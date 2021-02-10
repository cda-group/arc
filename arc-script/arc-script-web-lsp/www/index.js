import start from "arc-script-web-lsp"
// const { start } = require("../pkg/arc_script_web_lsp.js");

async function main() {
  await start(process.stdin, process.stdout);
}

main();
