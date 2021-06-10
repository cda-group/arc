-- Types (Entities)

type Person        = { name, email_address, credit_card, city, state }
type Item          = { description, name }
type OpenAuction   = { initial, reserve, expires }
type ClosedAuction = { date }
type Bid           = { item_id, bidder_id, time, price, date }
type Category      = { name, description }

-- Tasks (Business Objects)

task Map(f) (In) -> (Out) {
    on In(x) => emit Out(f(x))
}

task Auction() () -> () {
    
}

task Bid() () -> () {

}

extern fn dol_to_eur(dollar);

fn query1(bids: Stream[Bid]) {
    val people: Stream[Person] = source::People() ();
    val items: Stream[Item] = source::Items() ();
    val bids: Stream[Bid] = source::Bids() ();

    val mapped = Map(|bid| {
        item_id = bid.item_id,
        bidder_id = bid.bidder_id,
        eur = dol_to_eur(bid.price),
        time = bid.time
    }) (items);

    sink::Query1() (mapped)
}
