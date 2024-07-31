struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
        }
    }

    fn dot(&self, other: &Matrix) -> Matrix {
        let rows = self.data.len();
        let cols = other.data[0].len();
        let mut result = Matrix::new(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                for k in 0..self.data[0].len() {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    fn add(&self, other: &Matrix) -> Matrix {
        let rows = self.data.len();
        let cols = self.data[0].len();
        let mut result = Matrix::new(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
}




struct Embedding {
    vocab_size: usize,
    embedding_dim: usize,
    embeddings: Matrix,
}



impl Embedding {
    fn new(vocab_size: usize, embedding_dim: usize) -> Self {
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
        let mut result = Matrix::new(input.len(), self.embedding_dim);
        for (i, &token) in input.iter().enumerate() {
            result.data[i] = self.embeddings.data[token].clone();
        }
        result
    }
}





fn positional_encoding(seq_len: usize, embedding_dim: usize) -> Matrix {
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
        assert!(dim % heads == 0, "dim must be divisible by heads");
        let head_dim = dim / heads;
        let w_q = Matrix::new(dim, dim);
        let w_k = Matrix::new(dim, dim);
        let w_v = Matrix::new(dim, dim);
        let w_o = Matrix::new(dim, dim);
        MultiHeadAttention { heads, dim, head_dim, w_q, w_k, w_v, w_o }
    }

    fn forward(&self, query: &Matrix, key: &Matrix, value: &Matrix) -> Matrix {
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

        // Final linear layer
        concat_output.dot(&self.w_o)
    }
}








struct FeedForward {
    dim: usize,
    w1: Matrix,
    w2: Matrix,
    b1: Vec<f64>,
    b2: Vec<f64>,
}

impl FeedForward {
    fn new(dim: usize) -> Self {
        let w1 = Matrix::new(dim, dim);
        let w2 = Matrix::new(dim, dim);
        let b1 = vec![0.0; dim];
        let b2 = vec![0.0; dim];
        FeedForward { dim, w1, w2, b1, b2 }
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut output = Matrix::new(input.data.len(), self.dim);
        for i in 0..input.data.len() {
            for j in 0..self.dim {
                output.data[i][j] = 0.0;
                for k in 0..input.data[0].len() {
                    output.data[i][j] += input.data[i][k] * self.w1.data[k][j];
                }
                output.data[i][j] = (output.data[i][j] + self.b1[j]).max(0.0);
            }
        }

        let mut final_output = Matrix::new(output.data.len(), self.dim);
        for i in 0..output.data.len() {
            for j in 0..self.dim {
                final_output.data[i][j] = 0.0;
                for k in 0..output.data[0].len() {
                    final_output.data[i][j] += output.data[i][k] * self.w2.data[k][j];
                }
                final_output.data[i][j] += self.b2[j];
            }
        }
        final_output
    }
}

struct LayerNorm {
    dim: usize,
    gamma: Vec<f64>,
    beta: Vec<f64>,
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        let gamma = vec![1.0; dim];
        let beta = vec![0.0; dim];
        LayerNorm { dim, gamma, beta }
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let mut normed = Matrix::new(input.data.len(), self.dim);
        for i in 0..input.data.len() {
            let mean: f64 = input.data[i].iter().sum::<f64>() / self.dim as f64;
            let variance: f64 = input.data[i].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.dim as f64;
            for j in 0..self.dim {
                normed.data[i][j] = self.gamma[j] * (input.data[i][j] - mean) / (variance + 1e-6).sqrt() + self.beta[j];
            }
        }
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
        TransformerBlock {
            attention: MultiHeadAttention::new(heads, dim),
            feed_forward: FeedForward::new(dim),
            norm1: LayerNorm::new(dim),
            norm2: LayerNorm::new(dim),
        }
    }

    fn forward(&self, input: &Matrix) -> Matrix {
        let attention_output = self.attention.forward(input, input, input);
        let normed_attention_output = self.norm1.forward(&input.add(&attention_output));
        let feed_forward_output = self.feed_forward.forward(&normed_attention_output);
        self.norm2.forward(&normed_attention_output.add(&feed_forward_output))
    }
}

struct Transformer {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
}

impl Transformer {
    fn new(vocab_size: usize, embedding_dim: usize, num_blocks: usize, heads: usize) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            blocks.push(TransformerBlock::new(heads, embedding_dim));
        }
        Transformer {
            embedding: Embedding::new(vocab_size, embedding_dim),
            blocks,
        }
    }

    fn forward(&self, input: Vec<usize>) -> Matrix {
        let mut

 x = self.embedding.forward(input);
        x = x.add(&positional_encoding(x.data.len(), x.data[0].len()));

        for block in &self.blocks {
            x = block.forward(&x);
        }

        x
    }
}

fn main() {
    let vocab_size = 10000;
    let embedding_dim = 512;
    let num_blocks = 6;
    let heads = 8;

    let transformer = Transformer::new(vocab_size, embedding_dim, num_blocks, heads);

    let input = vec![1, 2, 3, 4, 5];
    let output = transformer.forward(input);

    for row in output.data {
        for value in row {
            print!("{:.4} ", value);
        }
        println!();
    }
}
