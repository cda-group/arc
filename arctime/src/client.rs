use kompact::prelude::*;

use std::sync::Arc;

use crate::control::*;
use crate::data::*;
use crate::pipeline::*;
use crate::port::*;
use crate::task::*;

#[derive(ComponentDefinition, Actor)]
pub struct Client {
    pub ctx: ComponentContext<Self>,
}

impl Client {
    pub(crate) fn new() -> Self {
        Self {
            ctx: ComponentContext::uninitialised(),
        }
    }
}

impl ComponentLifecycle for Client {}
