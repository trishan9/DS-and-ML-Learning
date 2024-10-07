___
## Day 3
### Topic: Multiple Linear Regression
*Date: October 5, 2024*

### Today's Learning Objectives Completed ✅
- Understanding Multiple Feature Linear Regression
- Vector notation in Linear Regression
- Feature representation and indexing
- Extended model equations

### Detailed Notes 📝

#### Multiple Features Introduction
![swappy-20241005_214718](https://github.com/user-attachments/assets/ae8f424d-5ad9-420a-aef7-e6a1bab277cf)


Important notation:
- n = number of features
- m = number of training examples
- x<sup>(i)</sup> = features of i<sup>th</sup> training example
- x<sub>j</sub><sup>(i)</sup> = value of feature j in i<sup>th</sup> training example

Example from the data:
- x<sup>(2)</sup> = [1416  3  2  40] (complete 2nd training example)
- x<sub>3</sub><sup>(2)</sup> = 2 (3rd feature of 2nd training example)

#### Model Extension
![swappy-20241005_214726](https://github.com/user-attachments/assets/8e3e9689-8c16-49a4-8e0d-ecd69540afb3)


Evolution of the model:
- Previously: f<sub>w,b</sub>(x) = wx + b
- Now with multiple features:
  ```
  fw,b(x) = w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄ + b
  ```

Example house price prediction:
```
fw,b(x) = 0.1x₁ + 4x₂ + 10x₃ - 2x₄ + 80
```
where:
- x₁: size in feet²
- x₂: number of bedrooms
- x₃: number of floors
- x₄: age of home in years
- b = 80: base price

#### Vector Notation
![swappy-20241005_214740](https://github.com/user-attachments/assets/eadb2826-0e00-4101-8f95-b40e344b7966)


Modern representation using vectors:
- w⃗ = [w₁ w₂ w₃ ... wₙ] (parameter vector)
- x⃗ = [x₁ x₂ x₃ ... xₙ] (feature vector)
- b is a single number (scalar)

Final model equation using dot product:
```
fw,b(x) = w⃗ · x⃗ + b = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

**Important Note**: This is multiple linear regression, not multivariate regression. The distinction is that we have multiple features (variables) but still predict a single output value.

#### Key Takeaways 🔑
1. Multiple features allow more complex and accurate predictions
2. Vector notation simplifies representation of multiple features
3. Dot product provides elegant mathematical formulation
4. Each feature has its own weight parameter (w)
5. Base price (b) remains a single scalar value

#### Practical Implementation Tips 💡
- Use vectors and matrices for efficient computation
- Keep track of feature indices carefully
- Document feature meanings and units
- Consider feature scaling for better performance
- Use proper indexing notation in code
