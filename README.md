
## Simple Programming Language

Made during the [ANU CSSA Hackathon in 2022](https://github.com/anucssa/hackathon-submissions-2022)

Dynamically typed, interpreted imperative language with very few constructs - no functions yet!

It has variables, lexically scoped if/else, while loops, arithmetic and boolean and string operators.

```
var i = 0;

while (i < 10) {
    print(i);
    i = i + 1
}
```

`cargo run file.lang` then gives:

```
0
1
2
3
4
5
6
7
8
9
```

:sparkles:
