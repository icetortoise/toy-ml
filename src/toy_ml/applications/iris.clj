(ns toy-ml.applications.iris
  (:require clojure.contrib.math))
(use '(toy-ml core mlp))
(use '(incanter core io stats))


;; load, randomize, train and test with confmat

;; todo: think about macros to help?


(def path-to-data
  "/Users/andywu/projects/toy-ml/src/toy_ml/applications/iris_proc.data")

(def ds (read-dataset path-to-data))

(defn- take-order [order]
  (fn [ds] (to-matrix ($ order :all ds))))

(defn randomize-to-matrix
  ([ds]
     (let [r-order (shuffle (range (nrow ds)))]
       ((take-order r-order) ds)))
  ([ds & others]
     (let [r-order (shuffle (range (nrow ds)))]
       (map (take-order r-order)
            (cons ds others)))))

(defn norm-column [col]
  (div (minus col (mean col))
       (- (apply max col)
          (apply min col))))


(defn normalize [m]
  (trans (matrix (map norm-column (trans m)))))

(defn- separate-index [groups n]
  (let [ng (div groups (sum groups))
        before-last (map (fn [x] (clojure.contrib.math/floor (* n x)))
                         (drop-last ng))
        lst (- n (sum before-last))
        index-sep (concat before-last [lst])]
    (next (reduce (fn [x y]
                    (concat x [(+ (last x) y)]))
                  (cons [0] index-sep)))))

(defn separate [groups ds]
  (let [seps (separate-index groups (nrow ds))]
    (map (fn [x y]
           ($ (range x y) :all ds))
         (cons 0 (drop-last seps))
         seps)))

(defn value-index-map [col]
  (loop [ele (first col)
         nxt (next col)
         ind 0
         result {}]
    (if (nil? ele) result
        (recur (first nxt) (next nxt) (inc ind)
               (assoc result ele ind)))))

(defn one-to-n [column]
  (let [values (apply sorted-set column)
        n (count values)]
    (matrix (map (fn [x]
                   (let [value-index (value-index-map values)]
                     (assoc (vec (repeat n 0))
                       (value-index x) 1)))
                 column))))

(defn run-iris [ds]
  (let [shuffled (randomize-to-matrix ds)
        origin-inputs ($ :all (range (dec (ncol shuffled)))
                         shuffled)
        origin-targets (one-to-n ($ :all (dec (ncol shuffled))
                                    shuffled))
        [train-inputs test-inputs]
        (separate [100 50] origin-inputs)
        [train-targets test-targets]
        (separate [100 50] origin-targets)
        weights (mlp-train train-inputs train-targets [3] 0.01 0.02
                           (make-logistic 1) (end-after-iter 1500))]
    (correct-percentage 
     (last (mlp-forward test-inputs weights (:forward (make-logistic 1))))
     test-targets)))
        
