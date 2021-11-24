use kompact::prelude::*;

use kompact::config_keys::system::LABEL;
use kompact::config_keys::system::THREADS;
use kompact::executors::crossbeam_workstealing_pool;

use omnipaxos::leader_election::ballot_leader_election::messages::BLEMessage;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatMsg;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatReply;
use omnipaxos::leader_election::ballot_leader_election::messages::HeartbeatRequest;
use omnipaxos::leader_election::ballot_leader_election::Ballot;
use omnipaxos::leader_election::ballot_leader_election::BallotLeaderElection;
use omnipaxos::messages::AcceptDecide;
use omnipaxos::messages::AcceptSync;
use omnipaxos::messages::Accepted;
use omnipaxos::messages::Decide;
use omnipaxos::messages::FirstAccept;
use omnipaxos::messages::Message;
use omnipaxos::messages::PaxosMsg;
use omnipaxos::messages::Prepare;
use omnipaxos::messages::Promise;
use omnipaxos::paxos::ProposeErr;
use omnipaxos::storage::memory_storage::MemorySequence;
use omnipaxos::storage::memory_storage::MemoryState;
use omnipaxos::utils::hocon_kv::CONFIG_ID;
use omnipaxos::utils::hocon_kv::HB_DELAY;
use omnipaxos::utils::hocon_kv::INITIAL_DELAY_FACTOR;
use omnipaxos::utils::hocon_kv::LOG_FILE_PATH;
use omnipaxos::utils::hocon_kv::PID;
use omnipaxos::utils::hocon_kv::PRIORITY;
use omnipaxos::utils::logger::create_logger;
type OmniPaxos<T> = omnipaxos::paxos::OmniPaxos<T, MemorySequence<T>, MemoryState>;

pub struct BallotPort;

impl Port for BallotPort {
    type Indication = Ballot;
    type Request = Never;
}
