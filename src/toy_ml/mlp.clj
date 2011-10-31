(ns toy-ml.mlp)
(use '(toy-ml core))
(use '(incanter core))

;; todo, linear/softmax...
;; todo, backward; momentem; confmax; early stop

(defn initial-weights [inputs targets hidden]
  (defn- rand-init [n] (- (rand (/ 2 (sqrt n))) (/ 1 (sqrt n))))
  (let [input_dim (+ 1 (ncol inputs))]
    [(m-map rand-init (matrix input_dim input_dim hidden))
     (m-map rand-init (matrix (+ 1 hidden) (+ 1 hidden)
                              (ncol targets)))]))

(defn intecept [inputs]
  (bind-columns (matrix -1 (nrow inputs) 1)
                inputs))

(defn make-logistic [beta]
  (fn [x] (/ 1 (+ 1 (exp (- (* beta x)))))))

(defn mlp-forward [inputs targets w_hidden w_out]
  (let [inputs (intecept inputs)
        logistic (make-logistic 1)
        activations (m-map logistic (mmult inputs w_hidden))
        outputs (m-map logistic (mmult (bind-columns
                                        (matrix -1 (nrow activations) 1)
                                        activations)
                                       w_out))]
    [activations outputs]))

(defn mlp-backward
  [inputs targets outputs hidden_activations w_hidden w_out]
  (let [delta_out (mult (minus targets outputs)
                        outputs
                        (minus 1 outputs))
        delta_hidden nil]
  ))

(defn mlp-train [inputs, targets, hidden, beta, momentum, out-type end-fn]
  (let [[weights_hidden weights_out]
        (initial-weights inputs targets hidden)]
    (loop [w_hidden weights_hidden
           w_out weights_out
           iter 1]
      (let [[outputs hidden_activations]
            (mlp-forward weights_hidden weights_out)
            [w_hidden_new w_out_new]
            (mlp-backward inputs targets outputs hidden_activations
                          w_hidden w_out)]
        (if (end-fn iter) [w_hidden_new w_out_new]
            (recur w_hidden_new w_out_new (+ 1 iter)))))))
      
