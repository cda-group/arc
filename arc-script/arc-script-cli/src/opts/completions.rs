use crate::opts::Completions;
use crate::opts::Opt;
use crate::opts::Shell;

use clap::App;
use clap::IntoApp;
use clap_generate::generate;
use clap_generate::generators::Bash;
use clap_generate::generators::Elvish;
use clap_generate::generators::Fish;
use clap_generate::generators::PowerShell;
use clap_generate::generators::Zsh;

impl Completions {
    /// Generates completions for the given shell.
    pub(crate) fn generate(&self) {
        let mut app: App = Opt::into_app().name("arc-script");
        let fd = &mut std::io::stdout();
        let name = "arc-script";
        match &self.shell {
            Shell::Bash => generate::<Bash, _>(&mut app, name, fd),
            Shell::Zsh => generate::<Zsh, _>(&mut app, name, fd),
            Shell::Fish => generate::<Fish, _>(&mut app, name, fd),
            Shell::Elvish => generate::<Elvish, _>(&mut app, name, fd),
            Shell::PowerShell => generate::<PowerShell, _>(&mut app, name, fd),
        }
    }
}
