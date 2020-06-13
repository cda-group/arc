import * as compiler from 'arc-script-wasm';
import {Terminal} from 'xterm';
import {FitAddon} from 'xterm-addon-fit';

var term = new Terminal({cursorBlink : true, rows : 50});
var elem = document.getElementById('terminal');
term.open(elem);
term.prompt = () => { term.write('\r\n$ '); };
term.writeln('This is a REPL for the arc-script language.');
term.prompt();
var curr_line = '';

term.onKey(e => {
  const printable = !e.domEvent.altKey && !e.domEvent.altGraphKey &&
                    !e.domEvent.ctrlKey && !e.domEvent.metaKey;

  // https://keycode.info/
  if (e.domEvent.keyCode === 13) {
    if (curr_line != '') {
      // <Enter>
      var input = curr_line;
      curr_line = '';
      term.writeln("");
      var msg = '';
      try {
        msg = compiler.compile(input).replace(/\n/g, '\n\r')
      } catch (err) {
        msg = err.message
      };
      term.write(msg);
      term.prompt();
    } else {
      term.prompt();
    }
  } else if (e.domEvent.keyCode === 8) {
    // <Backspace>
    if (term._core.buffer.x > 2) {
      term.write('\b \b');
      curr_line = curr_line.slice(0, -1);
    }
  } else if (e.domEvent.ctrlKey && e.domEvent.keyCode == 76) {
    // <C-L>
    term.clear()
  } else if (printable) {
    // <Key>
    curr_line += e.key;
    term.write(e.key);
  }
});

const fitAddon = new FitAddon();
term.loadAddon(fitAddon);
fitAddon.fit();
elem.focus();
term.focus();
