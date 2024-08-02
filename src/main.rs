#[derive(Clone)]
struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        println!("Creating new matrix with dimensions: {}x{}", rows, cols);
        Matrix {
            data: vec![vec![0.0; cols]; rows],
        }
    }

    fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.data[0].len(), other.data.len(), "Incompatible matrix dimensions for multiplication");
        
        let rows = self.data.len();
        let cols = other.data[0].len();
        println!("Performing matrix multiplication: {}x{} * {}x{}", rows, self.data[0].len(), other.data.len(), cols);
        let mut result = Matrix::new(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                for k in 0..other.data.len() {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    fn add(&self, other: &Matrix) -> Matrix {
        let rows = self.data.len();
        let cols = self.data[0].len();
        println!("Adding matrices: {}x{}", rows, cols);
        let mut result = Matrix::new(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    fn transpose(&self) -> Matrix {
        let rows = self.data[0].len();
        let cols = self.data.len();
        println!("Transposing matrix: {}x{} to {}x{}", cols, rows, rows, cols);
        let mut result = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    fn subtract(&self, other: &Matrix) -> Matrix {
        let rows = self.data.len();
        let cols = self.data[0].len();
        println!("Subtracting matrices: {}x{}", rows, cols);
        let mut result = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    fn mul_scalar(&self, scalar: f64) -> Matrix {
        let rows = self.data.len();
        let cols = self.data[0].len();
        println!("Multiplying matrix by scalar: {}x{} * {}", rows, cols, scalar);
        let mut result = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
}

fn softmax(input: &[f64]) -> Vec<f64> {
    println!("Applying softmax to vector of length {}", input.len());
    let max_val = input.iter().fold(input[0], |a, &b| if a > b { a } else { b });
    let exp_vals: Vec<f64> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp_vals: f64 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| x / sum_exp_vals).collect()
}

fn exp(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for i in 1..100 {
        term *= x / i as f64;
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    sum
}

fn ln(x: f64) -> f64 {
    let mut sum = 0.0;
    let mut term = (x - 1.0) / (x + 1.0);
    let mut n = 1.0;
    for _ in 0..100 {
        sum += term / n;
        term *= (x - 1.0) * (x - 1.0) / ((x + 1.0) * (x + 1.0));
        n += 2.0;
        if term.abs() < 1e-10 {
            break;
        }
    }
    2.0 * sum
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        println!("Initializing RNG with seed: {}", seed);
        Rng { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) ^ self.state
    }

    fn next_f64(&mut self) -> f64 {
        self.next() as f64 / u64::MAX as f64
    }
}

fn initialize_weights(matrix: &mut Matrix, rng: &mut Rng) {
    println!("Initializing weights for matrix: {}x{}", matrix.data.len(), matrix.data[0].len());
    for i in 0..matrix.data.len() {
        for j in 0..matrix.data[0].len() {
            matrix.data[i][j] = rng.next_f64() * 2.0 - 1.0;
        }
    }
}

struct Tokenizer {
    vocab: Vec<(String, usize)>,
}

impl Tokenizer {
    fn new() -> Self {
        println!("Creating new Tokenizer");
        Tokenizer { vocab: Vec::new() }
    }

    fn tokenize(&mut self, text: &str) -> Vec<usize> {
        println!("Tokenizing text of length: {}", text.len());
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_lowercase()).collect();
        let mut tokens = Vec::new();

        for word in words {
            let token = match self.vocab.iter().position(|(w, _)| w == &word) {
                Some(index) => index,
                None => {
                    let new_index = self.vocab.len();
                    self.vocab.push((word.clone(), new_index));
                    new_index
                }
            };
            tokens.push(token);
        }

        println!("Tokenized into {} tokens", tokens.len());
        tokens
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn decode(&self, token: usize) -> &str {
        &self.vocab[token].0
    }
}

struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    embeddings: Matrix,
}

impl Embedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        println!("Creating Embedding with vocab_size: {}, embedding_dim: {}", vocab_size, embedding_dim);
        let mut embeddings = Matrix::new(vocab_size, embedding_dim);
        let mut seed: u64 = 123456789;
        for i in 0..vocab_size {
            for j in 0..embedding_dim {
                embeddings.data[i][j] = Self::lcg_random(&mut seed);
            }
        }
        Embedding {
            vocab_size,
            embedding_dim,
            embeddings,
        }
    }

    fn lcg_random(seed: &mut u64) -> f64 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: u64 = 1 << 32;
        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % M;
        (*seed as f64) / (M as f64)
    }

    fn forward(&self, input: Vec<usize>) -> Matrix {
        println!("Embedding forward pass with input length: {}", input.len());
        let mut result = Matrix::new(input.len(), self.embedding_dim);
        for (i, &token) in input.iter().enumerate() {
            if token >= self.vocab_size {
                println!("Warning: token {} is out of vocabulary range", token);
                continue;
            }
            result.data[i] = self.embeddings.data[token].clone();
        }
        result
    }
}

fn positional_encoding(seq_len: usize, embedding_dim: usize) -> Matrix {
    println!("Generating positional encoding: seq_len={}, embedding_dim={}", seq_len, embedding_dim);
    let mut encoding = Matrix::new(seq_len, embedding_dim);
    for pos in 0..seq_len {
        for i in 0..embedding_dim {
            if i % 2 == 0 {
                encoding.data[pos][i] = (pos as f64 / 10000_f64.powf(i as f64 / embedding_dim as f64)).sin();
            } else {
                encoding.data[pos][i] = (pos as f64 / 10000_f64.powf((i - 1) as f64 / embedding_dim as f64)).cos();
            }
        }
    }
    encoding
}

struct MultiHeadAttention {
    heads: usize,
    dim: usize,
    head_dim: usize,
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
}

impl MultiHeadAttention {
    fn new(heads: usize, dim: usize) -> Self {
        println!("Creating MultiHeadAttention: heads={}, dim={}", heads, dim);
        assert!(dim % heads == 0, "dim must be divisible by heads");
        let head_dim = dim / heads;
        let w_q = Matrix::new(dim, dim);
        let w_k = Matrix::new(dim, dim);
        let w_v = Matrix::new(dim, dim);
        let w_o = Matrix::new(dim, dim);
        MultiHeadAttention { heads, dim, head_dim, w_q, w_k, w_v, w_o }
    }

    fn backward(&mut self, gradients: &Matrix, learning_rate: f64) -> Matrix {
        println!("MultiHeadAttention backward pass");
        let seq_len = gradients.data.len();
        
        // Backpropagate through w_o
        let d_concat = gradients.dot(&self.w_o.transpose());
        let mut d_w_o = Matrix::new(self.dim, self.dim);
        for i in 0..seq_len {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    d_w_o.data[j][k] += gradients.data[i][k] * d_concat.data[i][j];
                }
            }
        }
        self.w_o = self.w_o.subtract(&d_w_o.mul_scalar(learning_rate));

        // Split gradients for each head
        let mut d_heads = vec![Matrix::new(seq_len, self.head_dim); self.heads];
        for i in 0..seq_len {
            for h in 0..self.heads {
                for j in 0..self.head_dim {
                    d_heads[h].data[i][j] = d_concat.data[i][h * self.head_dim + j];
                }
            }
        }

        // Backpropagate through attention mechanism for each head
        let mut d_q = Matrix::new(seq_len, self.dim);
        let mut d_k = Matrix::new(seq_len, self.dim);
        let mut d_v = Matrix::new(seq_len, self.dim);

        for h in 0..self.heads {
            let d_head = &d_heads[h];
            
            // Compute gradients for q, k, v
            let mut d_q_head = Matrix::new(seq_len, self.head_dim);
            let mut d_k_head = Matrix::new(seq_len, self.head_dim);
            let mut d_v_head = Matrix::new(seq_len, self.head_dim);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    for k in 0..self.head_dim {
                        d_q_head.data[i][k] += d_head.data[i][k] * self.w_k.data[j][h * self.head_dim + k] / (self.head_dim as f64).sqrt();
                        d_k_head.data[j][k] += d_head.data[i][k] * self.w_q.data[i][h * self.head_dim + k] / (self.head_dim as f64).sqrt();
                        d_v_head.data[j][k] += d_head.data[i][k];
                    }
                }
            }

            // Accumulate gradients for q, k, v
            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    d_q.data[i][h * self.head_dim + j] = d_q_head.data[i][j];
                    d_k.data[i][h * self.head_dim + j] = d_k_head.data[i][j];
                    d_v.data[i][h * self.head_dim + j] = d_v_head.data[i][j];
                }
            }
        }

        // Update w_q, w_k, w_v
        self.w_q = self.w_q.subtract(&d_q.transpose().dot(gradients).mul_scalar(learning_rate));
        self.w_k = self.w_k.subtract(&d_k.transpose().dot(gradients).mul_scalar(learning_rate));
        self.w_v = self.w_v.subtract(&d_v.transpose().dot(gradients).mul_scalar(learning_rate));

        // Return gradients for the input
        d_q.add(&d_k).add(&d_v)
    }

fn forward(&self, query: &Matrix, key: &Matrix, value: &Matrix) -> Matrix {
        println!("MultiHeadAttention forward pass");
        let seq_len = query.data.len();
        
        // Project inputs to q, k, v
        let q = query.dot(&self.w_q);
        let k = key.dot(&self.w_k);
        let v = value.dot(&self.w_v);

        // Split heads
        let mut q_heads = Vec::new();
        let mut k_heads = Vec::new();
        let mut v_heads = Vec::new();

        for h in 0..self.heads {
            let start = h * self.head_dim;
            let end = start + self.head_dim;
            q_heads.push(Matrix { data: q.data.iter().map(|row| row[start..end].to_vec()).collect() });
            k_heads.push(Matrix { data: k.data.iter().map(|row| row[start..end].to_vec()).collect() });
            v_heads.push(Matrix { data: v.data.iter().map(|row| row[start..end].to_vec()).collect() });
        }

        // Compute attention for each head
        let mut head_outputs = Vec::new();
        for h in 0..self.heads {
            println!("Processing head {}", h);
            let q_head = &q_heads[h];
            let k_head = &k_heads[h];
            let v_head = &v_heads[h];

            // Compute attention scores
            let mut attention_scores = Matrix::new(seq_len, seq_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    for m in 0..self.head_dim {
                        attention_scores.data[i][j] += q_head.data[i][m] * k_head.data[j][m];
                    }
                    attention_scores.data[i][j] /= (self.head_dim as f64).sqrt();
                }
            }

            // Apply softmax
            for i in 0..seq_len {
                let mut max_val = attention_scores.data[i][0];
                for j in 1..seq_len {
                    if attention_scores.data[i][j] > max_val {
                        max_val = attention_scores.data[i][j];
                    }
                }
                let mut sum = 0.0;
                for j in 0..seq_len {
                    attention_scores.data[i][j] = (attention_scores.data[i][j] - max_val).exp();
                    sum += attention_scores.data[i][j];
                }
                for j in 0..seq_len {
                    attention_scores.data[i][j] /= sum;
                }
            }

            // Apply attention to values
            let mut head_output = Matrix::new(seq_len, self.head_dim);
            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    for k in 0..seq_len {
                        head_output.data[i][j] += attention_scores.data[i][k] * v_head.data[k][j];
                    }
                }
            }
            head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut concat_output = Matrix::new(seq_len, self.dim);
        for i in 0..seq_len {
            let mut idx = 0;
            for head_output in &head_outputs {
                for j in 0..self.head_dim {
                    concat_output.data[i][idx] = head_output.data[i][j];
                    idx += 1;
                }
            }
        }

        println!("MultiHeadAttention output shape: {}x{}", concat_output.data.len(), concat_output.data[0].len());

        // Final linear layer
        concat_output.dot(&self.w_o)
    }
}

struct FeedForward {
    input_dim: usize,
    output_dim: usize,
    w1: Matrix,
    w2: Matrix,
    b1: Vec<f64>,
    b2: Vec<f64>,
}

impl FeedForward {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        println!("Creating FeedForward: input_dim={}, output_dim={}", input_dim, output_dim);
        let w1 = Matrix::new(input_dim, input_dim * 4);
        let w2 = Matrix::new(input_dim * 4, output_dim);
        let b1 = vec![0.0; input_dim * 4];
        let b2 = vec![0.0; output_dim];
        FeedForward { input_dim, output_dim, w1, w2, b1, b2 }
    }


    fn backward(&mut self, gradients: &Matrix, learning_rate: f64) -> Matrix {
        println!("FeedForward backward pass");
        let mut hidden_gradients = Matrix::new(gradients.data.len(), self.input_dim * 4);
        let mut input_gradients = Matrix::new(gradients.data.len(), self.input_dim);

        // Backpropagate through second layer
        for i in 0..gradients.data.len() {
            for j in 0..self.input_dim * 4 {
                for k in 0..self.output_dim {
                    hidden_gradients.data[i][j] += gradients.data[i][k] * self.w2.data[j][k];
                    self.w2.data[j][k] -= learning_rate * gradients.data[i][k] * hidden_gradients.data[i][j];
                }
            }
        }

        // Update b2
        for k in 0..self.output_dim {
            for i in 0..gradients.data.len() {
                self.b2[k] -= learning_rate * gradients.data[i][k];
            }
        }

        // Backpropagate through first layer and ReLU
        for i in 0..gradients.data.len() {
            for j in 0..self.input_dim {
                for k in 0..self.input_dim * 4 {
                    if hidden_gradients.data[i][k] > 0.0 {
                        input_gradients.data[i][j] += hidden_gradients.data[i][k] * self.w1.data[j][k];
                        self.w1.data[j][k] -= learning_rate * hidden_gradients.data[i][k] * input_gradients.data[i][j];
                    }
                }
                self.b1[j] -= learning_rate * hidden_gradients.data[i][j];
            }
        }

        input_gradients
    }







    fn forward(&self, input: &Matrix) -> Matrix {
        println!("FeedForward forward pass");
        let mut hidden = Matrix::new(input.data.len(), self.input_dim * 4);
        for i in 0..input.data.len() {
            for j in 0..self.input_dim * 4 {
                hidden.data[i][j] = self.b1[j];
                for k in 0..self.input_dim {
                    hidden.data[i][j] += input.data[i][k] * self.w1.data[k][j];
                }
                hidden.data[i][j] = hidden.data[i][j].max(0.0); // ReLU activation
            }
        }

        let mut output = Matrix::new(input.data.len(), self.output_dim);
        for i in 0..input.data.len() {
            for j in 0..self.output_dim {
                output.data[i][j] = self.b2[j];
                for k in 0..self.input_dim * 4 {
                    output.data[i][j] += hidden.data[i][k] * self.w2.data[k][j];
                }
            }
        }
        println!("FeedForward output shape: {}x{}", output.data.len(), output.data[0].len());
        output
    }
}

struct LayerNorm {
    dim: usize,
    gamma: Vec<f64>,
    beta: Vec<f64>,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        println!("Creating LayerNorm: dim={}", dim);
        let gamma = vec![1.0; dim];
        let beta = vec![0.0; dim];
        LayerNorm { dim, gamma, beta }
    }




    fn backward(&mut self, gradients: &Matrix, learning_rate: f64) -> Matrix {
        println!("LayerNorm backward pass");
        let mut d_input = Matrix::new(gradients.data.len(), gradients.data[0].len());
        let mut d_gamma = vec![0.0; self.dim];
        let mut d_beta = vec![0.0; self.dim];

        for i in 0..gradients.data.len() {
            let mean: f64 = gradients.data[i].iter().sum::<f64>() / gradients.data[i].len() as f64;
            let variance: f64 = gradients.data[i].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / gradients.data[i].len() as f64;
            let std_dev = (variance + 1e-6).sqrt();


            for j in 0..gradients.data[i].len() {
                let x_centered = gradients.data[i][j] - mean;
                let x_norm = x_centered / std_dev;

                if j < self.dim {
                    d_gamma[j] += gradients.data[i][j] * x_norm;
                    d_beta[j] += gradients.data[i][j];
                }

                d_input.data[i][j] = if j < self.gamma.len() {
                    self.gamma[j] * (gradients.data[i][j] - (d_beta[j] + x_norm * d_gamma[j]) / gradients.data[i].len() as f64) / std_dev
                } else {
                    gradients.data[i][j]
                };
            }
        }

        // Update gamma and beta
        for j in 0..self.dim {
            self.gamma[j] -= learning_rate * d_gamma[j];
            self.beta[j] -= learning_rate * d_beta[j];
        }

        d_input
    }




    fn forward(&self, input: &Matrix) -> Matrix {
        println!("LayerNorm forward pass");
        let mut normed = Matrix::new(input.data.len(), self.dim);
        for i in 0..input.data.len() {
            let mean: f64 = input.data[i].iter().sum::<f64>() / self.dim as f64;
            let variance: f64 = input.data[i].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.dim as f64;
            for j in 0..self.dim {
                if j >= self.gamma.len() || j >= self.beta.len() {
                    println!("Warning: j={} is out of bounds for gamma/beta (len={})", j, self.gamma.len());
                    continue;
                }
                normed.data[i][j] = self.gamma[j] * (input.data[i][j] - mean) / (variance + 1e-6).sqrt() + self.beta[j];
            }
        }
        println!("LayerNorm output shape: {}x{}", normed.data.len(), normed.data[0].len());
        normed
    }
}

struct TransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    fn new(heads: usize, dim: usize) -> Self {
        println!("Creating TransformerBlock: heads={}, dim={}", heads, dim);
        TransformerBlock {
            attention: MultiHeadAttention::new(heads, dim),
            feed_forward: FeedForward::new(dim, dim),
            norm1: LayerNorm::new(dim),
            norm2: LayerNorm::new(dim),
        }
    }

    fn backward(&mut self, gradients: &Matrix, learning_rate: f64) -> Matrix {
        println!("TransformerBlock backward pass");
        // Backpropagate through feed forward layer
        let ff_gradients = self.feed_forward.backward(gradients, learning_rate);

        // Backpropagate through layer norm 2
        let norm2_gradients = self.norm2.backward(&ff_gradients, learning_rate);

        // Backpropagate through attention layer
        let attention_gradients = self.attention.backward(&norm2_gradients, learning_rate);

        // Backpropagate through layer norm 1
        let norm1_gradients = self.norm1.backward(&attention_gradients, learning_rate);

        norm1_gradients
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        println!("TransformerBlock forward pass");
        let attention_output = self.attention.forward(input, input, input);
        let normed_attention_output = self.norm1.forward(&input.add(&attention_output));
        let feed_forward_output = self.feed_forward.forward(&normed_attention_output);
        let output = self.norm2.forward(&normed_attention_output.add(&feed_forward_output));
        println!("TransformerBlock output shape: {}x{}", output.data.len(), output.data[0].len());
        output
    }
}

struct Transformer {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    output_layer: FeedForward,
}

impl Transformer {
    fn new(vocab_size: usize, embedding_dim: usize, num_blocks: usize, heads: usize) -> Self {
        println!("Creating Transformer: vocab_size={}, embedding_dim={}, num_blocks={}, heads={}", vocab_size, embedding_dim, num_blocks, heads);
        let mut rng = Rng::new(12345);  // Use a fixed seed for reproducibility
        let mut embedding = Embedding::new(vocab_size, embedding_dim);
        initialize_weights(&mut embedding.embeddings, &mut rng);

        let mut blocks = Vec::new();
        for i in 0..num_blocks {
            println!("Initializing TransformerBlock {}", i);
            let mut block = TransformerBlock::new(heads, embedding_dim);
            initialize_weights(&mut block.attention.w_q, &mut rng);
            initialize_weights(&mut block.attention.w_k, &mut rng);
            initialize_weights(&mut block.attention.w_v, &mut rng);
            initialize_weights(&mut block.attention.w_o, &mut rng);
            initialize_weights(&mut block.feed_forward.w1, &mut rng);
            initialize_weights(&mut block.feed_forward.w2, &mut rng);
            blocks.push(block);
        }

        let mut output_layer = FeedForward::new(embedding_dim, vocab_size);
        initialize_weights(&mut output_layer.w1, &mut rng);
        initialize_weights(&mut output_layer.w2, &mut rng);

        Transformer {
            embedding,
            blocks,
            output_layer,
        }
    }

fn forward(&self, input: &[usize]) -> Matrix {
        println!("Transformer forward pass");
        let mut x = self.embedding.forward(input.to_vec());
        println!("Embedded input shape: {}x{}", x.data.len(), x.data[0].len());
        x = x.add(&positional_encoding(x.data.len(), x.data[0].len()));
        println!("After positional encoding: {}x{}", x.data.len(), x.data[0].len());

        for (i, block) in self.blocks.iter().enumerate() {
            println!("Processing TransformerBlock {}", i);
            x = block.forward(&x);
            println!("After block {}: {}x{}", i, x.data.len(), x.data[0].len());
        }

        println!("Applying output layer");
        let output = self.output_layer.forward(&x);
        println!("Final output shape: {}x{}", output.data.len(), output.data[0].len());
        output
    }

    fn train(&mut self, input: &[usize], target: &[usize], learning_rate: f64) -> f64 {
        println!("Training on input of length {}", input.len());
        let output = self.forward(input);
        let mut loss = 0.0;

        let mut gradients = Matrix::new(output.data.len(), output.data[0].len());
        for i in 0..output.data.len() {
            let target_index = if i < target.len() { target[i] } else { 0 };
            let probs = softmax(&output.data[i]);
            for j in 0..output.data[0].len() {
                gradients.data[i][j] = probs[j] - if j == target_index { 1.0 } else { 0.0 };
                if j == target_index {
                    loss -= probs[j].ln();
                }
            }
        }

        println!("Calculated loss: {}", loss);

        // Backpropagate through output layer
        println!("Backpropagating through output layer");
        let output_gradients = self.output_layer.backward(&gradients, learning_rate);

        // Backpropagate through transformer blocks
        println!("Backpropagating through transformer blocks");
        let mut block_gradients = output_gradients;
        for (i, block) in self.blocks.iter_mut().enumerate().rev() {
            println!("Backpropagating through block {}", i);
            block_gradients = block.backward(&block_gradients, learning_rate);
        }

        // Update embedding layer
        println!("Updating embedding layer");
        for i in 0..input.len() {
            if i >= self.embedding.embeddings.data.len() {
                println!("Warning: input token {} is out of embedding range", input[i]);
                continue;
            }
            for j in 0..self.embedding.embedding_dim {
                if j >= block_gradients.data[0].len() {
                    println!("Warning: j={} is out of range for block_gradients", j);
                    continue;
                }
                self.embedding.embeddings.data[input[i]][j] -= learning_rate * block_gradients.data[i][j];
            }
        }

        loss
    }
}





fn main() {
    println!("Starting main function");
    // Read the text file
    let contents = include_str!("../Heany.txt");
    println!("Read file contents, length: {}", contents.len());

    // Tokenize the text
    let mut tokenizer = Tokenizer::new();
    let tokens = tokenizer.tokenize(&contents);
    println!("Tokenized text, number of tokens: {}", tokens.len());

    // Define training parameters
    let seq_length = 40;
    let epochs = 4;
    let learning_rate = 0.01;

    let total_iterations = epochs * (tokens.len() - seq_length - 1);
    let mut current_iteration = 0;

    // Initialize the transformer
    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 128;
    let num_blocks = 3;
    let heads = 4;

    println!("Initializing transformer with vocab_size={}, embedding_dim={}, num_blocks={}, heads={}", 
             vocab_size, embedding_dim, num_blocks, heads);
    let mut transformer = Transformer::new(vocab_size, embedding_dim, num_blocks, heads);

    println!("Starting training loop: seq_length={}, epochs={}, learning_rate={}", seq_length, epochs, learning_rate);
    for epoch in 0..epochs {
        println!("Starting epoch {}", epoch + 1);
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        for i in 0..tokens.len() - seq_length - 1 {
            let input = &tokens[i..i+seq_length];
            let target = &tokens[i+1..i+seq_length+1];
            

            current_iteration += 1;
            let progress = (current_iteration as f64 / total_iterations as f64) * 100.0;
            println!("Progress: {:.2}% ({}/{})", progress, current_iteration, total_iterations);

            let loss = transformer.train(input, target, learning_rate);

            total_loss += loss;
            batch_count += 1;
            if batch_count % 100 == 0 {
                println!("Epoch {}, Batch {}: Average Loss = {}", epoch + 1, batch_count, total_loss / batch_count as f64);
            }
        }
        println!("Epoch {} completed, Average Loss: {}", epoch + 1, total_loss / batch_count as f64);
    }


    // Generate predictions
    let prompt = "Between my finger and my thumb";
    println!("Generating predictions for prompt: '{}'", prompt);
    let mut input_tokens = tokenizer.tokenize(prompt);
    println!("Tokenized prompt: {:?}", input_tokens);

    for i in 0..10 {
        println!("Generating token {}", i + 1);
        let output = transformer.forward(&input_tokens);
        if output.data.is_empty() {
            println!("Error: Empty output from transformer");
            break;
        }
        let probs = softmax(&output.data.last().unwrap());
        let next_token = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        let next_word = tokenizer.decode(next_token);
        print!("{} ", next_word);

        input_tokens.push(next_token);
        input_tokens = input_tokens[1..].to_vec();
    }
    println!();
    println!("Prediction generation completed");

}
