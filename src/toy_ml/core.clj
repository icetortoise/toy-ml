(ns toy-ml.core)
(use '(incanter core))

(defn- try-matrix [r]
  (if (sequential? r)
    (matrix r)
    r))
(defn m-map
  "A map function returns a matrix. The matrix-map function
returns a sequence and will lost information when it is
applied to a vector."
  ([f m]
     (if (= (nrow m) 1)
       (trans (matrix (matrix-map f m)))
       (let [r (matrix-map f m)]
         (try-matrix r))))
  ([f m & ms]
     (if (= (nrow m) 1)
       (trans (matrix (apply matrix-map f m ms)))
       (let [r (apply matrix-map f m ms)]
         (try-matrix r)))))

(defn end-after-iter [iter]
  (fn [n] (if (>= n iter) true false)))

(defn sum-correct [x]
  (defn- sum-scalar-or-seq [x]
    (if (sequential? x) (sum x) x))
  (if (matrix? x)
    (let [row-sum (reduce plus x)]
      (sum-scalar-or-seq row-sum))))
