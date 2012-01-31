(ns toy-ml.boost
  (:use [incanter core io])
  (:require [incanter.stats :as stats]))


(declare grids-along grid-with-min-error)
(defn- lable [threshold x]
    (if (> x threshold)
      1 -1))

(defn- init-weights [n]
  (repeat n (/ 1 n)))

(defn- grids-along
  "int -> int -> dataset -> (float)"
  [axis n-grid ds]
  (let [col ($ axis ds)
        minimum (apply min col)
        maximum (apply max col)]
    (range minimum maximum (/ (- maximum minimum)
                              n-grid))))

(defn- grid-with-min-error
  "(float) -> [label] -> (float) -> float"
  [ds-col grid-coll targets weights]
  (defn- error [grid]
    (let [class-coll (map #(lable grid %)
                          ds-col)
          indicators (map #(if (= %1 %2) 0 1)
                          class-coll targets)]
      (sum ($= indicators * weights))))
  (let [err-coll (map error grid-coll)
        err-grid-map (zipmap err-coll grid-coll)]
    (-> err-grid-map
        sort first)))
  
(defn split-along-axis-train
  "dataset -> (label) -> (float) -> int -> (dataset -> classified result)"
  [ds targets weights axis
   {:keys [n-grid] :or {n-grid 10}}]
  (let [grids (grids-along axis n-grid ds)
        [error threshold] (grid-with-min-error ($ axis ds) grids targets weights)]
    {:pred-fn (fn [ds] (map #(lable threshold %)
                               ($ axis ds)))
     :grid threshold
     :error error}
     ))

(defn- uniform-samples [[nrow ncol]]
  "generating demo dataset"
  (to-dataset (repeatedly nrow #(stats/sample-uniform ncol))))

(defn- classify-samples [ds]
  "positive if both dimention is larger then 0.4."
  ($map #(if (and (> %1 0.4)
                  (> %2 0.4)) 1 -1)
        [:col-0 :col-1] ds))

(defn- make-boostable-trainer [n-grid]
  (fn [ds targets weights]
    (split-along-axis-train ds targets weights (rand-nth (range (ncol ds))) n-grid)))

(defn- alpha [error]
  (log (/ (- 1 error) error)))

(defn- normalize [xs]
  ($= xs / (sum xs)))

(defn- new-weights [weights pred-fn alpha ds targets]
  (normalize ($= weights *
                 (exp ($= alpha * (map #(if (= %1 %2) 0 1) (pred-fn ds) targets))))))

(defn- make-boosted [alphas classifiers]
  (fn [ds]
    (let [predicted (matrix (map #(% ds) classifiers))]
      (map #(if (< % 0) -1 1) ($= (trans (matrix alphas)) <*> predicted)))))

(defn boost [ds targets trainer]
  "dataset -> [label] -> (dataset -> [label] -> weights -> classifier)
  -> classified result"
  (loop [weights (init-weights (nrow ds))
         alphas []
         under-classifiers []
         iter 1]
    (let [classifier (trainer ds targets weights)]
      (if (or (> iter 100) (> (:error classifier) 0.5))
        {:alphas alphas
         :under-classifiers under-classifiers
         :boosted (make-boosted alphas under-classifiers)}
        (let [alp (alpha (:error classifier))]
          (recur (new-weights weights (:pred-fn classifier)
                              alp ds targets)
                 (conj alphas alp)
                 (conj under-classifiers (:pred-fn classifier))
                 (+ 1 iter)))))))
