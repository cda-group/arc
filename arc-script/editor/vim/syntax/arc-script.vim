" Vim syntax file
" Language: arc-script

if version < 600
  syntax clear
elseif exists("b:current_syntax")
  finish
endif

runtime! syntax/rust.vim syntax/rust/*.vim

let b:current_syntax = "arc-script"

"=============================================================================
" Overrides
"=============================================================================
hi clear rustMacroVariable
syn clear rustMacroVariable
syn match rustSigil /&\s\+[&~@*$][^)= \t\r\n]/me=e-1,he=e-1  display
syn match rustSigil /[&~@*$][^)= \t\r\n]/me=e-1,he=e-1  display
"=============================================================================
" Operators
"=============================================================================
syn keyword arcOperator <-
syn keyword arcOperator :=
syn keyword arcOperator <>
syn keyword arcOperator \|>
syn keyword arcOperator ???
syn keyword arcOperator ;;
syn keyword arcOperator **
syn keyword arcOperator ~
syn keyword arcOperator $
syn keyword arcOperator and
syn keyword arcOperator band
syn keyword arcOperator bor
syn keyword arcOperator bxor
syn keyword arcOperator is
syn keyword arcOperator or
syn keyword arcOperator unwrap
syn keyword arcOperator enwrap
syn keyword arcOperator xor
hi def link arcOperator Operator
"=============================================================================
" Statements
"=============================================================================
syn keyword arcStatement fun
syn keyword arcStatement task
syn keyword arcStatement on
syn keyword arcStatement emit
syn keyword arcStatement log
syn keyword arcStatement exit
syn keyword arcStatement after
syn keyword arcStatement every
syn keyword arcStatement val
syn keyword arcStatement var
" Not statements, but looks nice syntactically
syn keyword arcStatement not
syn keyword arcStatement in
syn keyword arcStatement is
hi def link arcStatement Statement
"=============================================================================
" Conditionals
"=============================================================================
syn keyword arcConditional by
hi def link arcConditional Conditional
"=============================================================================
" Reserved Keywords
"=============================================================================
syn keyword arcKeyword end,
syn keyword arcKeyword of
syn keyword arcKeyword shutdown
syn keyword arcKeyword sink
syn keyword arcKeyword source
syn keyword arcKeyword then
syn keyword arcKeyword where
hi def link arcKeyword Keyword
"=============================================================================
" Primitive Types
"=============================================================================
syn keyword arcType bf16
syn keyword arcType f16
syn keyword arcType unit
hi def link arcType Type
"=============================================================================
" Floats
"=============================================================================
syn match arcFloat "[[:digit:]]\+\.[[:digit:]]f16" display
syn match arcFloat "[[:digit:]]\+\.[[:digit:]]bf16" display
hi def link arcFloat Float
"=============================================================================
" Constants
"=============================================================================
syn match arcConstant "[[:digit:]]\+ns"
syn match arcConstant "[[:digit:]]\+us"
syn match arcConstant "[[:digit:]]\+ms"
syn match arcConstant "[[:digit:]]\+s"
syn match arcConstant "[[:digit:]]\+m"
syn match arcConstant "[[:digit:]]\+h"
syn match arcConstant "[[:digit:]]\+d"
syn match arcConstant "[[:digit:]]\+w"
hi def link arcConstant Constant
"=============================================================================
" Comments
"=============================================================================
syn match arcComment "#.*"
hi def link arcComment Comment
