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

(defn max-val-index [coll]
  (reduce 
   (fn [[val-1 pos-1] [val-2 pos-2]]
     (if (< val-1 val-2) [val-2 pos-2]
         [val-1 pos-1]))
   (partition 2 (interleave
                 coll
                 (range (count coll))))))
                     

(defn- conf-row [row]
  (if (sequential? row)
    (let [[val index] (max-val-index row)]
      (assoc (vec (repeat (count row) 0))
        index 1))
    (recur [row (- 1 row)])))

(defn confmat [outputs targets]
  (let [output-classified (matrix (map conf-row outputs))
        targets-classified (matrix (map conf-row targets))]
    (mmult (trans output-classified)
           targets-classified)))

(defn correct-percentage [outputs targets]
  (let [confmat (confmat outputs targets)]
    (* 100
       (/ (trace confmat) (sum-correct confmat)))))
