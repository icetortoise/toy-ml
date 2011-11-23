(ns toy-ml.core
  (:require clojure.contrib.math))
(use '(incanter core stats))

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

;; a repeat-until-convergence func/macro?
(defn end-after-iter [iter]
  (fn [_ _ n] (if (>= n iter) true false)))

(def default-iter 10000)

(defn end-when-cost-converge-below [threshold]
  (fn [cur prev iter]
    (let [ratio (abs (/ (- prev cur) prev))]
      (if (or (= 0 prev) (= iter default-iter)) true
          (if (<= ratio threshold)
            true false)))))

(defn sum-correct [x]
  (defn- sum-scalar-or-seq [x]
    (if (sequential? x) (sum x) x))
  (if (matrix? x)
    (let [row-sum (reduce plus x)]
      (sum-scalar-or-seq row-sum))))

;(defmacro with-analysis-infor)


;; ------data preparation and normalization
;; ----------------------------------------
(defn max-val-index [coll]
  (reduce 
   (fn [[val-1 pos-1] [val-2 pos-2]]
     (if (< val-1 val-2) [val-2 pos-2]
         [val-1 pos-1]))
   (partition 2 (interleave
                 coll
                 (range (count coll))))))

(defn- take-order [order]
  (fn [ds] ($ order :all ds)))

(defn randomize
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

;; ------ test framework to run and evaluate algorithms and
;; ------ automatically select parameters

;; until-convergence with-params ...

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

(defn unreg-cost [outputs targets]
  (/ (/ (sum-correct (mult (minus outputs targets) (minus outputs targets)))
        2)        
     (nrow outputs)))
