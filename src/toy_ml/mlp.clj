(ns toy-ml.mlp)
(use '(toy-ml core))
(use '(incanter core))

;; todo, momentem; confmax; early stop or regularization?


(defn initial-weights [inputs targets hidden]
  (defn- rand-init [n] (- (rand (/ 2 (sqrt n))) (/ 1 (sqrt n))))
  (defn- each-adj-pair [col]
    (drop-last (map (fn [x y] [x y])
                    col (conj (vec (rest col)) (first col)))))
  (defn- make-random-matrix [[n1 n2]]
    (m-map rand-init (matrix (+ 1 n1) (+ 1 n1) n2)))
  (let [nnodes (flatten [(ncol inputs)
                         hidden
                         (ncol targets)])]
    (map make-random-matrix (each-adj-pair nnodes))))

(defn intecept [inputs]
  (bind-columns (matrix -1 (nrow inputs) 1)
                inputs))

(defn unintecept [x]
  (trans (rest (trans x))))

(defn make-logistic [beta]
  {:forward (fn [x] (/ 1 (+ 1 (exp (- (* beta x))))))
   :backward (fn [targets outputs]
               (mult (minus targets outputs)
                        outputs
                        (minus 1 outputs)))})

(defn make-linear []
  {:forward identity
   :backward (fn [targets outputs]
               (div (minus targets outputs)
                    (nrow outputs)))})

;; skipping softmax implementation
(comment (defn make-softmax []
  {:forward identity
  :backward (fn [targets outputs]
               (div (minus targets outputs)
                    (nrow outputs)))}))

(defn mlp-forward [inputs weights forward-fn]
  (defn- compute-activations [previous weight]
    (conj previous
          (m-map forward-fn
                 (mmult (intecept (last previous))
                        weight))))
  (rest (reduce compute-activations
                (cons [inputs] weights))))

(defn- hidden-act-weight-pairs
  [acts weights]
  (reverse
   (map (fn [act weight]
          [act weight])
        (drop-last acts)
        (rest weights))))

(defn- h-delta [prev-deltas [act weight]]
  (conj prev-deltas
        (unintecept (mult (intecept act) (minus 1 (intecept act))
                          (mmult (last prev-deltas) (trans weight))))))

(defn- new-weight [weight delta input-or-hidden-act]
  (plus weight
        (mmult (trans (intecept input-or-hidden-act))
               delta)))

(defn mlp-backward
  ([inputs targets activations weights backward-fn]
     (let [delta-out (backward-fn targets (last activations))
           h-deltas (reduce h-delta
                            (cons [delta-out]
                                  (hidden-act-weight-pairs
                                   activations weights)))]
       (map new-weight
            weights
            (reverse (cons delta-out h-deltas))
            (cons inputs (drop-last activations))))))

(defn mlp-train [inputs targets hidden momentum act-fns end-fn]
  (let [weights (initial-weights inputs targets hidden)]
    (loop [weights weights
           inputs inputs
           targets targets
           iter 1]
      (let [activations
            (mlp-forward inputs weights (:forward act-fns))
            new-weights
            (mlp-backward inputs targets activations
                          weights (:backward act-fns))]
        (if (end-fn iter) new-weights
            (let [new_order (shuffle (range (nrow inputs)))]
              (recur new-weights
                     ($ new_order :all inputs)
                     ($ new_order :all targets)
                     (+ 1 iter))))))))
