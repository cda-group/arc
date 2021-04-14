# Stateful tasks

A **state variable** is a variable which lives in the context of a task. State variables are initialized when creating a task (i.e., when all parameters including streams have been passed to the task).

```text
Item ::=
  | 'task' Name '(' (Name ':' Type ',')* ')' ':' Interface '->' Interface '{' TaskItem* '}'
  | ..

TaskItem ::=
  | 'var' Name (':' Type)? '=' Expr ';'
  | 'val' Name (':' Type)? '=' Expr ';'
  | ..
```
