# JavaScript  notes

- JavaScript (TS) or TypeScript (JS)?
    - TypeScript is object-oriented and strongly typed, but it also comprises to JS code. TS has static typing, i.e., the compiler enforces variables to maintain the same type, which allows type errors to be captured at compile time so there are less run time errors. 
    - Here is a famous JS typing glitch
      ```jsx
      userInput = "2"  // (a string)
      // if we compare it with an int 
      userInput == 2
      // we will get true, but if used in arithmetics
      console.log(userInput + 2);
      // Expectation: 4
      // Output: 22
      ```
    - Here are some good articles -
    - [https://radixweb.com/blog/typescript-vs-javascript](https://radixweb.com/blog/typescript-vs-javascript)
    - [https://www.section.io/engineering-education/typescript-static-typing/](https://www.section.io/engineering-education/typescript-static-typing/)

- HTML and CSS refresher
    - [https://www.freecodecamp.org/learn/](https://www.freecodecamp.org/learn/)

- JS ES6  
  (these are the notes I've taken reviewing ES6 standard)
    - use *let* and *const* to avoid a repeated variable declaration
    - undeclared local will be automatically declared globally
    - ===, !== **strictly** equality/inequality operator in JS (no type conversion)
    - The *object literal* in JS holds a mixture of dictionary, list, like json containers. It can also hold nested objects and nested arrays, and access via dot or bracket notations.
    - the conditional operator is very similar to C++
    - the chaining of multiple conditional (ternary) operators is an interesting design but I’m not sure if it provides better readability
    - *var* declared variable can be mutated globally, while let is scope limited
    - *Arrow function* provides a syntactic sugar for anonymous functions, it provides simple interface but I’m no sure
    - Use Destructuring Assignment to Assign Variables from Nested Objects
        
        ```jsx
        const user = {
          johnDoe: { 
            age: 34,
            email: 'johnDoe@freeCodeCamp.com'
          }
        };
        
        // Which one is more readable?
        userAge = user.johnDoe.age
        userEmail = user.johnDoe.email
        // or destructuring assignment?
        const { johnDoe: { age: userAge, email: userEmail }} = user;
        
        ```
        
    - *template literal* is similar to the f-string in Python
    - asynchronous! Promise!

- D3.js for data visualization
  - in progress