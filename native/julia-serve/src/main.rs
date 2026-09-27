//! julia-serve: self-contained HTTP server for julia-system-one.
//!
//! ONE binary per platform (linux/win/mac x arm64/x64, musl = zero deps).
//! Embeds: ONNX Runtime (statically linked via `ort`), HF tokenizers
//! (pure Rust), Axum HTTP server. The JS side only spawns this binary
//! and proxies `/v1/systemone` over localhost HTTP.
//!
//! Wire protocol: 100% compatible with TypeSafe Jev (/v1/systemone).

mod prompt;
mod schema;

use std::path::PathBuf;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::Json, routing::post, Router};
use anyhow::anyhow;
use clap::Parser;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;
use tracing::info;

use prompt::PromptBuilder;
use schema::*;

#[derive(Parser, Debug)]
#[command(name = "julia-serve", about = "Self-contained julia-system-one inference server")]
struct Args {
    /// Model directory (model.onnx, tokenizer.json, rl_agent_config.json)
    #[arg(long, default_value = "models")]
    model_dir: PathBuf,
    /// Bind host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Bind port (0 = pick free port, prints it for the spawner)
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Intra-op threads (0 = ORT default)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Bearer API key (optional; also JULIA_API_KEY)
    #[arg(long)]
    api_key: Option<String>,
    /// Token budget for the state (overrides rl_agent_config.json).
    /// The config ships a conservative 2048 so weak machines stay usable;
    /// raise it for long documents. Cost follows the input's real length,
    /// not this value, so short inputs are unaffected.
    #[arg(long)]
    max_len: Option<usize>,
}

struct AppState {
    // ort::Session needs &mut per run and is not Sync: guard with Mutex.
    // Inference is sequential (batch=1 loop), same as the JS engine.
    session: std::sync::Mutex<Session>,
    prompts: std::sync::Mutex<PromptBuilder>,
    max_len: usize,
    head_max_len: usize,
    temperatures: Vec<f32>,
    api_key: Option<String>,
}

/// Rewrite the symbolic dimensions of the ONNX graph to concrete values.
///
/// The Julia-1 export leaves dimensions as expressions - "6*batch",
/// "batch*tokens", "(tokens//batch)" - where the earlier checkpoints had plain
/// names. ONNX Runtime accepts those symbolically and resolves them its own
/// way, which turned out to disagree with the wasm (tract) path on the same
/// input: same model, same question, different answer with high confidence on
/// both sides. Forcing the dimensions makes both runtimes see the same graph.
///
/// An expression we do not recognise is left alone deliberately: a wrong
/// substitution would produce wrong numbers silently.
fn patch_dim_params(raw: &[u8], s: usize, m: usize) -> anyhow::Result<Vec<u8>> {
    use prost::Message as _;
    use tract_onnx::pb as onnx_pb;
    use onnx_pb::tensor_shape_proto::dimension::Value as V;

    let mut model: onnx_pb::ModelProto = onnx_pb::ModelProto::decode(raw)?;
    let mut fix = |dims: &mut Vec<onnx_pb::tensor_shape_proto::Dimension>| {
        for d in dims.iter_mut() {
            if let Some(V::DimParam(p)) = &mut d.value {
                let resolved = match p.as_str() {
                    "batch" | "batch_size" => Some("1".to_string()),
                    "seq_len" | "tokens" => Some(s.to_string()),
                    "num_markers" | "options" => Some(m.to_string()),
                    "6*batch" => Some("6".to_string()),
                    "batch*tokens" | "(tokens//batch)" => Some(s.to_string()),
                    _ => None,
                };
                if let Some(v) = resolved {
                    *p = v;
                }
            }
        }
    };
    if let Some(g) = model.graph.as_mut() {
        for vi in g.input.iter_mut().chain(g.output.iter_mut()).chain(g.value_info.iter_mut()) {
            if let Some(t) = vi.r#type.as_mut() {
                if let Some(onnx_pb::type_proto::Value::TensorType(tt)) = &mut t.value {
                    if let Some(sh) = tt.shape.as_mut() {
                        fix(&mut sh.dim);
                    }
                }
            }
        }
    }
    let mut buf = Vec::with_capacity(raw.len());
    model.encode(&mut buf)?;
    Ok(buf)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let api_key = args.api_key.or_else(|| std::env::var("JULIA_API_KEY").ok());

    // ---- ONNX Runtime init (ort 2.x: init returns () once committed) ----
    ort::init().with_name("julia-serve").commit();

    // ort::Error is not Send/Sync-compatible with anyhow's `?` in this
    // context: map to strings explicitly.
    fn oe<T, E: std::fmt::Debug>(r: Result<T, E>) -> anyhow::Result<T> {
        r.map_err(|e| anyhow::anyhow!("{e:?}"))
    }
    let model_path = args.model_dir.join("model.onnx");
    info!("loading {}", model_path.display());

    // ---- Config (read first: the dimension patch needs the token budgets) ----
    let cfg_path = args.model_dir.join("rl_agent_config.json");
    let (cfg_max_len, head_max_len, temperatures) = read_config(&cfg_path);
    // explicit flag > JULIA_MAX_LEN > config file
    let env_max_len = std::env::var("JULIA_MAX_LEN").ok().and_then(|v| v.parse::<usize>().ok());
    let max_len = args
        .max_len
        .or(env_max_len)
        .filter(|n| *n > 0)
        .unwrap_or(cfg_max_len);
    tracing::info!("max_len = {max_len}");
    // ORT 1.23 (the last Intel dylib, used by the mac-x64-legacy feature)
    // predates ORT_ENABLE_LAYOUT/ORT_ENABLE_ALL - max valid is EXTENDED.
    // Level2 already covers the fusions that matter for a CPU transformer
    // (GELU, LayerNorm, Attention), so the legacy build uses it.
    // NOTE: if the 1.22 optimizer itself segfaults on this graph, drop to
    // Disabled via the JULIA_ORT_NO_OPTIMIZE env (CI diagnosis knob).
    #[cfg(feature = "mac-x64-legacy")]
    let opt_level = if std::env::var("JULIA_ORT_NO_OPTIMIZE").is_ok() {
        GraphOptimizationLevel::Disable
    } else {
        GraphOptimizationLevel::Level2
    };
    #[cfg(not(feature = "mac-x64-legacy"))]
    let opt_level = GraphOptimizationLevel::Level3;
    // Apply the dimension patch, exactly as the wasm path does. Julia-1's
    // export leaves dimensions as expressions ("6*batch", "(tokens//batch)")
    // and the two runtimes resolve them differently unless the graph is made
    // concrete first: without this, x64 answered "sales" where arm64 answered
    // "tech" on the same input.
    let raw = std::fs::read(&model_path)?;
    let model_bytes = patch_dim_params(&raw, max_len, head_max_len)?;

    // Pin the CPU execution provider explicitly.
    //
    // Left to itself, ONNX Runtime picks the first registered provider, and on
    // Windows the ort-sys prebuilt is the DirectML build - so the same model
    // answered differently on Windows than on Linux (3/10 vs 10/10 on the same
    // questions). This package is a CPU decision engine; the provider must not
    // depend on which prebuilt happened to be downloaded.
    let mut builder = oe(Session::builder()?.with_optimization_level(opt_level))?;
    if args.threads > 0 {
        builder = oe(builder.with_intra_threads(args.threads))?;
    }
    // with_execution_providers returns BuilderResult, not Result
    let mut builder = builder
        .with_execution_providers([ort::execution_providers::CPU::default().build()])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let session = oe(builder.commit_from_memory(&model_bytes))?;

    // ---- Tokenizer (pure Rust, reads tokenizer.json directly) ----
    let tok_path = args.model_dir.join("tokenizer.json");
    let tokenizer =
        Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let prompts = PromptBuilder::new(
        tokenizer.clone(),
        mask_token_id(&tokenizer),
        cls_token_id(&tokenizer),
        sep_token_id(&tokenizer),
    );

    let state = Arc::new(AppState {
        session: std::sync::Mutex::new(session),
        prompts: std::sync::Mutex::new(prompts),
        max_len,
        head_max_len,
        temperatures,
        api_key,
    });

    let app = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/health", axum::routing::get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", args.host, args.port)).await?;
    let addr = listener.local_addr()?;
    // Machine-readable line for the JS spawner.
    println!("JULIA_READY {addr}");
    info!("listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "backend": "julia-serve" }))
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Result<Json<OutBody>, (StatusCode, Json<serde_json::Value>)> {
    // Auth (optional)
    if let Some(key) = &st.api_key {
        let ok = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|v| v == format!("Bearer {key}"))
            .unwrap_or(false);
        if !ok {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Unauthorized" })),
            ));
        }
    }
    let bytes = axum::body::to_bytes(req.into_body(), 4 * 1024 * 1024)
        .await
        .map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "error": format!("read body: {e}") })),
            )
        })?;
    let body: InBody = serde_json::from_slice(&bytes).map_err(|e| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": format!("invalid body: {e}") })),
        )
    })?;

    if body.questions.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({ "error": "missing state/questions" })),
        ));
    }

    // NOTE: Session is not Sync; run inference on a blocking thread with a
    // per-request &mut borrow via Mutex. Throughput: sequential, matches
    // current JS behavior (batch=1 loop). A pool comes later if needed.
    let st2 = st.clone();
    let out = tokio::task::spawn_blocking(move || infer_all(&st2, &body))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("task: {e}") })),
            )
        })??;
    Ok(Json(out))
}

fn infer_all(st: &AppState, body: &InBody) -> Result<OutBody, (StatusCode, Json<serde_json::Value>)> {
    use ort::value::Tensor;

    let model_name = body.model.clone().unwrap_or_else(|| "julia-1".into());
    let mut answers = serde_json::Map::new();
    let mut total_in = 0usize;

    for (qid, qdef) in &body.questions {
        let qtype = qdef.qtype.as_str();
        let internal = prompt::InternalQ::from_def(qid, qdef);
        let (ids, markers) = {
            let mut prompts = st.prompts.lock().unwrap();
            prompts.build(&body.state, &internal, st.max_len, st.head_max_len)
        };
        total_in += ids.len();

        let s = ids.len();
        let m = markers.len();
        let ids_i64: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
        let attn = vec![1i64; s];
        let mp: Vec<i64> = markers.iter().map(|&v| v as i64).collect();
        let mm: Vec<bool> = vec![true; m];
        let qt: Vec<i64> = vec![match qtype {
            "choice" => 0,
            "score" => 1,
            _ => 2,
        }];

        // Session needs &mut: serialize through the Mutex (batch=1 loop,
        // same as the JS engine).
        let err500 = |what: &str, e: std::fmt::Arguments| -> (StatusCode, Json<serde_json::Value>) {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("{what}: {e:?}") })),
            )
        };
        let logits: Vec<f32> = {
            let mut session = st.session.lock().unwrap();
            let t_ids = Tensor::from_array((vec![1, s], ids_i64.into_boxed_slice()))
                .map_err(|e| err500("input_ids", format_args!("{e:?}")))?;
            let t_attn = Tensor::from_array((vec![1, s], attn.into_boxed_slice()))
                .map_err(|e| err500("attention_mask", format_args!("{e:?}")))?;
            let t_mp = Tensor::from_array((vec![1, m], mp.into_boxed_slice()))
                .map_err(|e| err500("marker_pos", format_args!("{e:?}")))?;
            let t_mm = Tensor::from_array((vec![1, m], mm.into_boxed_slice()))
                .map_err(|e| err500("marker_mask", format_args!("{e:?}")))?;
            let t_qt = Tensor::from_array((vec![1], qt.into_boxed_slice()))
                .map_err(|e| err500("qtype", format_args!("{e:?}")))?;
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => t_ids,
                    "attention_mask" => t_attn,
                    "marker_pos" => t_mp,
                    "marker_mask" => t_mm,
                    "qtype" => t_qt,
                ])
                .map_err(|e| err500("infer", format_args!("{e:?}")))?;
            let (_shape, data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| err500("logits", format_args!("{e:?}")))?;
            data.to_vec()
        };

        let row = &logits[..m.min(logits.len())];
        let temp = st
            .temperatures
            .get(match qtype {
                "choice" => 0,
                "score" => 1,
                _ => 2,
            })
            .copied()
            .unwrap_or(1.0);
        answers.insert(qid.clone(), prompt::decode(&internal, qdef, row, temp));
    }

    Ok(OutBody {
        model: model_name,
        answers: serde_json::Value::Object(answers),
        usage: Usage {
            input_tokens: total_in,
            output_tokens: body.questions.len() * 4,
        },
    })
}

fn read_config(path: &std::path::Path) -> (usize, usize, Vec<f32>) {
    let (mut max_len, mut head_max, mut temps) = (1024usize, 256usize, vec![1.0, 1.0, 1.0]);
    if let Ok(raw) = std::fs::read_to_string(path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(n) = v.get("max_len").and_then(|x| x.as_u64()) {
                max_len = n as usize;
            }
            if let Some(n) = v.get("head_max_len").and_then(|x| x.as_u64()) {
                head_max = n as usize;
            }
            if let Some(t) = v.get("temperature").and_then(|x| x.as_array()) {
                let ts: Vec<f32> = t.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect();
                if ts.len() == 3 {
                    temps = ts;
                }
            }
        }
    }
    (max_len, head_max, temps)
}

fn mask_token_id(t: &Tokenizer) -> u32 {
    t.token_to_id("<mask>").unwrap_or(4)
}
fn cls_token_id(t: &Tokenizer) -> u32 {
    t.token_to_id("<bos>").unwrap_or(2)
}
fn sep_token_id(t: &Tokenizer) -> u32 {
    t.token_to_id("<eos>").unwrap_or(1)
}
