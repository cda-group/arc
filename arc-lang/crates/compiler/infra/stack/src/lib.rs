#![allow(unused)]

#[derive(Clone, Debug)]
pub struct Stack<K, E, T>(Vec<Scope<K, E, T>>);

pub type Name = String;

#[derive(Clone, Debug)]
pub struct Scope<K, E, T> {
    pub kind: K,
    pub expr_namespace: Namespace<E>,
    pub type_namespace: Namespace<T>,
}

impl<K, E, T> Scope<K, E, T> {
    fn new(kind: K) -> Self {
        Scope {
            kind,
            expr_namespace: Namespace::new(),
            type_namespace: Namespace::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Namespace<D>(Vec<(Name, D)>);

impl<D> Namespace<D> {
    pub fn new() -> Self {
        Self(vec![])
    }

    fn bind(&mut self, x: Name, decl: D) {
        self.0.push((x, decl));
    }

    pub fn find(&self, x0: &Name) -> Option<&D> {
        self.0
            .iter()
            .rev()
            .find_map(|(x1, decl)| if x0 == x1 { Some(decl) } else { None })
    }

    fn find_mut(&mut self, x0: &Name) -> Option<&mut D> {
        self.0
            .iter_mut()
            .rev()
            .find_map(|(x1, decl)| if x0 == x1 { Some(decl) } else { None })
    }

    pub fn iter(&self) -> impl Iterator<Item = &(Name, D)> {
        self.0.iter().rev()
    }
}

impl<K, E: Clone, T: Clone> Stack<K, E, T> {
    pub fn new(kind: K) -> Self {
        Self(vec![Scope::new(kind)])
    }

    pub fn depth(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Scope<K, E, T>> {
        self.0.iter().rev()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Scope<K, E, T>> {
        self.0.iter_mut().rev()
    }

    pub fn current(&mut self) -> &mut K {
        &mut self.0.last_mut().unwrap().kind
    }

    pub fn push_scope(&mut self, kind: K) {
        self.0.push(Scope::new(kind));
    }

    pub fn pop_scope(&mut self) -> K {
        self.0.pop().expect("ICE: Tried to pop empty scope").kind
    }

    pub fn bind_expr_decl(&mut self, x: Name, decl: E) {
        self.0.last_mut().unwrap().expr_namespace.bind(x, decl);
    }

    pub fn bind_type_decl(&mut self, x: Name, decl: T) {
        self.0.last_mut().unwrap().type_namespace.bind(x, decl);
    }

    pub fn find_expr_decl(&self, x0: &Name) -> Option<E> {
        self.iter().find_map(|s| s.expr_namespace.find(x0).cloned())
    }

    pub fn find_type_decl(&self, x0: &Name) -> Option<T> {
        self.iter().find_map(|s| s.type_namespace.find(x0).cloned())
    }

    pub fn find_expr_decl_mut(&mut self, x0: &Name) -> Option<&mut E> {
        self.iter_mut().find_map(|s| s.expr_namespace.find_mut(x0))
    }

    pub fn find_type_decl_mut(&mut self, x0: &Name) -> Option<&mut T> {
        self.iter_mut().find_map(|s| s.type_namespace.find_mut(x0))
    }
}
