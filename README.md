
## Simple Programming Language

Made during the [ANU CSSA Hackathon in 2022](https://github.com/anucssa/hackathon-submissions-2022)

This is a very small dynamically typed, interpreted, imperative language

It has variables, lexically scoped if/else, functions, while loops, arithmetic and boolean and string operators.

```
var i = 0;

while (i < 10) {
    print(i);
    i = i + 1
};

fn fact (n) {
    if (n == 0)
        1
    else
        n * fact(n-1)
};
print fact(6)
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
720
```

:sparkles:
