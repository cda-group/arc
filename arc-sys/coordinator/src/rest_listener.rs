use super::server::Server;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Json;
use axum::Router;
use shared::api::CoordinatorAPI;
use std::net::SocketAddr;

#[derive(Clone)]
struct Actor {
    server: Server,
}

impl Actor {
    pub fn new(server: Server) -> Self {
        Self { server }
    }

    async fn handle(self: State<Self>, Json(msg): Json<CoordinatorAPI>) -> impl IntoResponse {
        match msg {
            CoordinatorAPI::Shutdown => {
                self.server.shutdown().await;
                StatusCode::OK.into_response()
            }
            _ => StatusCode::BAD_REQUEST.into_response(),
        }
    }
}

pub struct RestListener;

impl RestListener {
    pub fn start(port: u16, server: Server) {
        tokio::spawn(async move {
            let router = Router::new()
                .route("/", post(Actor::handle))
                .with_state(Actor::new(server));

            tracing::info!("Listening for clients on {}", port);

            axum::Server::bind(&SocketAddr::from(([127, 0, 0, 1], port)))
                .serve(router.into_make_service())
                .await
                .unwrap();
        });
    }
}
