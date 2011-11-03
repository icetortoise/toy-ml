(ns toy-ml.mlp)
(use '(toy-ml core))
(use '(incanter core))

;; todo, linear/softmax...
;; todo, predict, momentem; confmax; early stop

(defn initial-weights [inputs targets hidden]
  (defn- rand-init [n] (- (rand (/ 2 (sqrt n))) (/ 1 (sqrt n))))
  (let [input_dim (+ 1 (ncol inputs))]
    [(m-map rand-init (matrix input_dim input_dim hidden))
     (m-map rand-init (matrix (+ 1 hidden) (+ 1 hidden)
                              (ncol targets)))]))

(defn intecept [inputs]
  (bind-columns (matrix -1 (nrow inputs) 1)
                inputs))

(defn unintecept [x]
  (trans (rest (trans x))))

(defn make-logistic [beta]
  (fn [x] (/ 1 (+ 1 (exp (- (* beta x)))))))

(defn mlp-forward [inputs w_hidden w_out beta]
  (let [inputs (intecept inputs)
        logistic (make-logistic beta)
        activations (m-map logistic (mmult inputs w_hidden))
        outputs (m-map logistic (mmult (intecept activations)
                                       w_out))]
    [activations outputs]))

(defn mlp-backward
  [inputs targets outputs hidden_act w_hidden w_out]
  (let [delta_out (mult (minus targets outputs)
                        outputs
                        (minus 1 outputs))
        delta_hidden (mult (intecept hidden_act) (minus 1 (intecept hidden_act))
                           (mmult delta_out (trans w_out)))]
    [(plus w_hidden (mmult (trans (intecept inputs)) (unintecept delta_hidden)))
     (plus w_out (mmult (trans (intecept hidden_act)) delta_out))]
  ))

(defn mlp-train [inputs targets hidden beta momentum out-type end-fn]
  (let [[weights_hidden weights_out]
        (initial-weights inputs targets hidden)]
    (loop [w_hidden weights_hidden
           w_out weights_out
           inputs inputs
           targets targets
           iter 1]
      (let [[activations outputs]
            (mlp-forward inputs w_hidden w_out beta)
            [w_hidden_new w_out_new]
            (mlp-backward inputs targets outputs activations
                          w_hidden w_out)]
        (if (end-fn iter) [w_hidden_new w_out_new]
            (let [new_order (shuffle (range (nrow inputs)))]
              (recur w_hidden_new w_out_new
                     ($ new_order :all inputs)
                     ($ new_order :all targets)
                     (+ 1 iter))))))))
