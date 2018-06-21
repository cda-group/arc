grammar Arc;

import ArcLexerRules;

r  : 'hello' TIdentifier '!'
 	| TBoolLit '?'
 	| (TI8Lit|TF32Lit) TPlus TPlus;         // match keyword hello followed by an identifier
