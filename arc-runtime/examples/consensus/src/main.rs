#![allow(unused)]

mod ble;
mod config;
mod omni;
mod port;
mod state;
mod system;

use crate::config::Config;
use crate::state::State;
use crate::system::System;

use kompact::prelude::*;
use omnipaxos::leader_election::ballot_leader_election::Ballot;
use omnipaxos::storage::Entry;
use rand::Rng;

use arc_runtime::prelude::*;

fn main() {
    let config = Config::load("omnipaxos.conf").unwrap();
    let ble_initial_leader = None;

    let system = System::new(
        config.num_nodes,
        config.num_threads,
        config.ble_hb_delay,
        config.ble_initial_delay_factor,
        ble_initial_leader,
    );

    let node = system.nodes.get(&1).unwrap();

    let (ble_promise, ble_future) = promise();
    node.ble
        .on_definition(|c| c.asks.push_back(Ask::new(ble_promise, ())));

    system.start_all_nodes();

    let elected_leader = ble_future.wait_timeout(config.wait_timeout).unwrap();

    let mut proposal_node: u64;

    loop {
        proposal_node = rand::thread_rng().gen_range(1..=config.num_nodes as u64);

        if proposal_node != elected_leader.pid {
            break;
        }
    }

    let node = system.nodes.get(&proposal_node).unwrap();

    let (omni_promise, omni_future) = promise();
    node.omni.on_definition(|c| {
        c.asks.push_back(Ask::new(omni_promise, ()));
        c.propose_normal(State::new("abc".to_owned(), 123)).unwrap();
    });

    omni_future.wait_timeout(config.wait_timeout).unwrap();

    system.kompact.shutdown();
}
