(ns toy-ml.core)
(use '(incanter core))

(defn m-map
  "A map function returns a matrix. The matrix-map function
returns a sequence and will lost information when it is
applied to a vector."
  ([f m]
     (if (= (nrow m) 1)
       (trans (matrix (matrix-map f m)))
       (matrix (matrix-map f m))))
  ([f m & ms]
     (if (= (nrow m) 1)
       (trans (matrix (apply matrix-map f m ms)))
       (matrix (apply matrix-map f m ms)))))