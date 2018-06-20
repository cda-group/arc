grammar Arc;

import ArcLexerRules;

r  : 'hello' TIdentifier '!'
 	| TBoolLit '?';         // match keyword hello followed by an identifier
