(ns toy-ml.applications.pnoz
  (:require clojure.contrib.math))
(use '(toy-ml core mlp))
(use '(incanter core io stats charts))

(def path-to-data
  "/Users/andywu/projects/toy-ml/src/toy_ml/applications/PNoz.dat")


(def ds (read-dataset path-to-data :delim \space))

(def t 2) (def k 3)

(defn time-serialize [col t k]
  (seq (loop [c col
              result '()]
         (let [seris (take (inc k) (take-nth t c))]
           (if (or (nil? c)
                   (not (= (inc k) (count seris))))
             result
             (recur (next c)
                    (concat result [seris])))))))

(defn input-target-matrix [time-seris]
  (let [input-col
        (map (fn [x] (drop-last x))
             time-seris)
        target-col
        (map (fn [x] (last x))
             time-seris)]
    [(matrix input-col) (matrix target-col)]))

(defn prepare [dsm t k]
  (input-target-matrix
   (time-serialize (norm-column ($ :all 2 dsm)) t k)))

(defn sep-to-train-test [inputs targets]
  (let [[train test] (separate [2449 400] inputs)
        [train-t test-t] (separate [2449 400] targets)
        [random-train random-target]
        (randomize train train-t)]
    [random-train random-target test test-t]))

(defn run-pnoz [ds]
  (let [dsm (to-matrix ds)
        [inputs targets] (prepare dsm t k)
        [train train-t test test-t] (sep-to-train-test inputs targets)
        weights (mlp-train train train-t [3] 0.03 1 (make-linear) (end-after-iter 500))]
    [(last (mlp-forward test weights (:forward (make-linear))))
     test-t]))

(defn result-plot [[out target]]
  (doto (scatter-plot (range (count out))
                      out)
        (add-points (range (count target))
                      target)))
    