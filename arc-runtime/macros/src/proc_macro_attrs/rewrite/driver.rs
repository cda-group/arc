use proc_macro as pm;

#[cfg(feature = "legacy")]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    quote::quote!(#item).into()
}

#[cfg(not(feature = "legacy"))]
pub(crate) fn rewrite(_attr: syn::AttributeArgs, item: syn::ItemFn) -> pm::TokenStream {
    use crate::new_id;
    let block = &item.block;
    let id = item.sig.ident;
    let component_id = new_id(format!("{}Component", id));
    let run_id = new_id(format!("{}_run", id));

    quote::quote! (

        #[derive(ComponentDefinition, Actor)]
        struct #component_id {
            ctx: ComponentContext<Self>,
        }

        impl #component_id {
            fn new() -> Self {
                Self {
                    ctx: ComponentContext::uninitialised()
                }
            }
        }

        #[rewrite]
        fn #run_id() {
            #block
        }

        impl ComponentLifecycle for #component_id {
            fn on_start(&mut self) -> Handled {
                let component = self.ctx().component();
                let mutator = instantiate_immix(ImmixOptions::default());
                let ctx = Context::new(component, mutator);
                call!(#run_id());
                self.ctx().system().shutdown_async();
                Handled::DieNow
            }
        }

        fn #id() {
            let system = KompactConfig::default().build().unwrap();
            let component = system.create(move || #component_id::new());
            system.start(&component);
            system.await_termination();
        }
    )
    .into()
}
