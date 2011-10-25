(ns toy-ml.linear)
(use '(incanter core))

(defn normalize [input]
  (bind-rows (matrix -1 1 (ncol input))
             input))

(defn rand-matrix
  ([n nrow ncol]
     (rand-matrix n (matrix 0 nrow ncol)))
  ([n mat]
     (matrix (matrix-map (fn [x] (rand 1)) mat))))

(defn end-after-iter [iter]
  (fn [n] (if (>= n iter) true false)))

(defn activate [val]
  (if (> val 0) 1 0)
  )

(defn pcn
  "Takes input matrix(each colume is a data point) and target matrix(each column is a taget vector),
learning rate and a function indicating when to stop learning, return the learned weights."
  [input target l-rate end-fn]
  (let [normalized (normalize input)
        weights (rand-matrix 1 (nrow normalized)
                              (nrow target))]
    (loop [w weights iter 1]
      (let [ys (trans (matrix (matrix-map activate (mmult (trans w) normalized))))] ;; ugly...fix!
        (if (end-fn iter) w
            (recur (plus w
                         (mult l-rate
                               (mmult normalized
                                      (trans (minus target ys)))))
                   (+ iter 1)))))))

(defn predict [input weights]
  (activate (mmult (trans weights) (normalize input))))

;; (pcn (trans (matrix [[0 0] [0 1] [1 0] [1 1]]))
;;		    (trans (matrix [0 1 1 1])) 0.1 (end-after-iter 1000))