(ns toy-ml.mlp)
(use '(toy-ml core))
(use '(incanter core))

;; todo: softmax
(defn initial-weights [inputs targets hidden]
  (defn- rand-init [n] (- (rand (/ 2 (sqrt n))) (/ 1 (sqrt n))))
  (defn- each-adj-pair [col]
    (drop-last (map (fn [x y] [x y])
                    col (conj (vec (rest col)) (first col)))))
  (defn- make-random-matrix [[n1 n2]]
    (m-map rand-init (matrix (+ 1 n1) (+ 1 n1) n2)))
  (let [nnodes (flatten [(ncol inputs)
                         (filter pos? (filter integer? hidden))
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
;; Use 1-to-N encoding for multiclass for simplicity
(comment (defn make-softmax []
  {:forward identity
  :backward (fn [targets outputs]
               (div (minus targets outputs)
                    (nrow outputs)))}))

(defn mlp-forward [inputs weights forward-fn]
  (defn- compute-h-act [previous weight]
    (conj previous
          (m-map (:forward (make-logistic 1))
                 (mmult (intecept (last previous))
                        weight))))
  (let [h-acts (rest (reduce compute-h-act
                             (cons [inputs] (drop-last weights))))
        next-to-outputs (or (last h-acts) inputs)]
    (concat h-acts
            [(m-map forward-fn
                    (mmult (intecept next-to-outputs)
                           (last weights)))])))

(def mlp-recall (comp last mlp-forward))

(defn cost [outputs targets weights reg-coff]
  (/ (+ (/ (sum-correct (mult (minus outputs targets) (minus outputs targets)))
           2)
        (* (sum (map (fn[w] (sum-correct (mult w w))) weights)) reg-coff))
     (nrow outputs)))

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

(defn- new-weight-fn [reg-coff l-rate n-inputs]
  (defn- bind-zero-first-row [m]
    (bind-rows (matrix 0 1 (ncol m))
               (rest m)))
  (fn [weight delta input-or-hidden-act]
    (plus weight
          (mult l-rate
                (minus (mmult (trans (intecept input-or-hidden-act))
                              delta)
                       (mult reg-coff
                             (div (bind-zero-first-row weight) n-inputs)))))))

(defn mlp-backward
  ([inputs targets activations weights backward-fn reg-coff l-rate]
     (let [delta-out (backward-fn targets (last activations))
           h-deltas (reduce h-delta
                            (cons [delta-out]
                                  (hidden-act-weight-pairs
                                   activations weights)))]
       (map (new-weight-fn reg-coff l-rate (nrow inputs))
            weights
            (reverse (cons delta-out h-deltas))
            (cons inputs (drop-last activations))))))

(defn mlp-single-iter [inputs targets reg-coff l-rate act-fns weights]
  (let [activations
        (mlp-forward inputs weights (:forward act-fns))
        new-weights
        (mlp-backward inputs targets activations
                      weights (:backward act-fns)
                      reg-coff l-rate)]
    [(last activations) new-weights]))

;; todo: return weights and costs
(defn mlp-train [inputs targets hidden reg-coff l-rate act-fns end-fn]
  (let [weights (initial-weights inputs targets hidden)]
    (loop [weights weights
           inputs inputs
           targets targets
           iter 1
           cst-vec [100000]]
      (let [[outputs new-weights]
            (mlp-single-iter inputs targets reg-coff l-rate act-fns weights)
            new-cst (unreg-cost outputs targets)]
        (if (= 1 (mod iter 100)) (println (last cst-vec) new-cst))
        (if (end-fn new-cst cst-vec iter) [new-weights (conj cst-vec new-cst)]
            (let [new_order (shuffle (range (nrow inputs)))]
              (recur new-weights
                     ($ new_order :all inputs)
                     ($ new_order :all targets)
                     (+ 1 iter)
                     (conj cst-vec new-cst))))))))

(defn choose-params [inputs targets inputs-val targets-val]      
  (let [trained (for [hidden [[3] [4]]
                      reg-coff [0.01 0.03]
                      l-rate [0.02 0.06]
                      act-fns [(make-logistic 1)]
                      end-fn [(end-after-iter 900)]]
                  (let [[weights costs]
                        (mlp-train inputs targets hidden reg-coff l-rate
                                   act-fns end-fn)
                        outputs-val (last (mlp-forward inputs-val weights
                                                       (:forward (make-logistic 1))))]
                    {:score (correct-percentage outputs-val targets-val)
                     :weights weights
                     :hidden hidden
                     :reg-off reg-coff
                     :l-rate l-rate
                     :recall-fn (let [weights weights
                                      out-fn (:forward (make-logistic 1))]
                                  (fn [inputs]
                                    (mlp-forward inputs weights out-fn)))}))]
    trained))


