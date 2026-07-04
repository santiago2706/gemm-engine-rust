use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use tiny_http::{Header, Method, Request, Response, Server};

use crate::engine::{self, BenchResult};

const STATIC_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/visual-demo");
const ADDRESS: &str = "127.0.0.1:7878";

pub fn run() {
    let server = match Server::http(ADDRESS) {
        Ok(server) => server,
        Err(err) => {
            println!("No se pudo iniciar el servidor en {}: {}", ADDRESS, err);
            return;
        }
    };

    println!("\nServidor web activo en http://{}", ADDRESS);
    println!("Abre esa dirección en tu navegador para ver la demo con benchmarks reales.");
    println!("Presiona Ctrl+C en esta terminal para detener el servidor.\n");

    for request in server.incoming_requests() {
        handle_request(request);
    }
}

fn handle_request(request: Request) {
    if request.method() != &Method::Get {
        let _ = request.respond(Response::from_string("Method Not Allowed").with_status_code(405));
        return;
    }

    let url = request.url().to_string();
    let (path, query) = split_query(&url);

    match path {
        "/api/benchmark" => {
            let params = parse_query(query);
            let strategy = params.get("strategy").map(String::as_str).unwrap_or("base");
            let n = matrix_size_from(&params);
            let result = engine::run_benchmark(strategy, n);
            respond_json(request, &bench_result_json(&result, None));
        }
        "/api/benchmark/all" => {
            let params = parse_query(query);
            let n = matrix_size_from(&params);

            let base = engine::run_benchmark("base", n);
            let base_checksum = base.checksum;
            let simd = engine::run_benchmark("simd", n);
            let cache = engine::run_benchmark("cache", n);
            let parallel = engine::run_benchmark("parallel", n);

            let body = format!(
                "[{},{},{},{}]",
                bench_result_json(&base, Some(base_checksum)),
                bench_result_json(&simd, Some(base_checksum)),
                bench_result_json(&cache, Some(base_checksum)),
                bench_result_json(&parallel, Some(base_checksum)),
            );
            respond_json(request, &body);
        }
        _ => serve_static(request, path),
    }
}

fn matrix_size_from(params: &HashMap<String, String>) -> usize {
    params
        .get("n")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256)
        .clamp(16, 1536)
}

fn bench_result_json(result: &BenchResult, base_checksum: Option<f64>) -> String {
    let matches_field = match base_checksum {
        Some(bc) => format!(",\"matches_base\":{}", (bc - result.checksum).abs() < 1e-1),
        None => String::new(),
    };
    format!(
        "{{\"strategy\":\"{}\",\"n\":{},\"elapsed_ms\":{:.4},\"gflops\":{:.4},\"checksum\":{:.4}{}}}",
        result.strategy, result.n, result.elapsed_ms, result.gflops, result.checksum, matches_field
    )
}

fn respond_json(request: Request, body: &str) {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json; charset=utf-8"[..]).unwrap();
    let _ = request.respond(Response::from_string(body).with_header(header));
}

fn serve_static(request: Request, path: &str) {
    let file_name = if path == "/" { "index.html" } else { path.trim_start_matches('/') };
    let full_path: PathBuf = [STATIC_DIR, file_name].iter().collect();

    match fs::read(&full_path) {
        Ok(bytes) => {
            let header = Header::from_bytes(&b"Content-Type"[..], content_type_for(file_name).as_bytes()).unwrap();
            let _ = request.respond(Response::from_data(bytes).with_header(header));
        }
        Err(_) => {
            let _ = request.respond(Response::from_string("404 Not Found").with_status_code(404));
        }
    }
}

fn content_type_for(file_name: &str) -> &'static str {
    if file_name.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if file_name.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if file_name.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

fn split_query(url: &str) -> (&str, &str) {
    match url.find('?') {
        Some(idx) => (&url[..idx], &url[idx + 1..]),
        None => (url, ""),
    }
}

fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            let key = parts.next()?.to_string();
            let value = parts.next().unwrap_or("").to_string();
            Some((key, value))
        })
        .collect()
}
