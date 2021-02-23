use arcon::prelude::*;

/// An operator for converting one stream into another.
#[derive(Default)]
pub struct Convert<I, O> {
    _marker: std::marker::PhantomData<(I, O)>,
}

impl<I, O> Convert<I, O> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

pub trait ConvertStream<O: ArconType> {
    fn convert(self) -> Stream<O>;
}

impl<I: ArconType, O: ArconType> ConvertStream<O> for Stream<I>
where
    I: Into<O>,
{
    fn convert(self) -> Stream<O> {
        self.operator(OperatorBuilder {
            constructor: Arc::new(|_| Convert::new()),
            conf: Default::default(),
        })
    }
}

impl<I: ArconType, O: ArconType> Operator for Convert<I, O>
where
    I: Into<O>,
{
    type IN = I;
    type OUT = O;
    type TimerState = ArconNever;
    type OperatorState = ();

    fn handle_element(
        &mut self,
        element: ArconElement<Self::IN>,
        mut ctx: OperatorContext<Self, impl Backend, impl ComponentDefinition>,
    ) -> OperatorResult<()> {
        let ArconElement { timestamp, data } = element;
        let data = data.into();
        let element = ArconElement { timestamp, data };
        ctx.output(element);
        Ok(())
    }

    arcon::ignore_timeout!();
    arcon::ignore_persist!();
}
