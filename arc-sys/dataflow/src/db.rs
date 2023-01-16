use futures::Future;

#[derive(Clone)]
pub enum Database {
    #[cfg(feature = "tikv")]
    Remote(tikv_client::RawClient),
    Local(sled::Db),
}

pub fn block_on<F: Future<Output = T>, T>(f: F) -> T {
    tokio::runtime::Runtime::new()
        .expect("failed to create runtime")
        .block_on(f)
}

impl Database {
    #[cfg(feature = "tikv")]
    pub fn remote(addr: &str) -> Self {
        let db = block_on(async {
            let client = tikv_client::RawClient::new(vec![addr])
                .await
                .expect("Failed to connect to TiKV");
            client
        });
        Self::Remote(db)
    }

    pub fn local(path: &str) -> Self {
        let db = sled::open(path).expect("Failed to connect to sled");
        Self::Local(db)
    }
}
