(ns toy-ml.dtree
  (:use [toy-ml core]
        [incanter core io]
        [clojure.core.match :only [match]]))

(defn entropy [distribution]
  (sum (map #(if (= 0 %) 0
                 (- ($= % * (log2 %))))
            distribution)))

(defn gini [distribution]
  (- 1 (sum (map #($= % ** 2)
                 distribution))))

(defn- seq-made-sure [x]
  (if (sequential? x) x
      [x]))

(defn data-gain [ds target gain-fn]
  (let [target-count-coll
        (seq-made-sure ($ :target-count
                          ($rollup :count :target-count
                                   target ds)))
        s (sum target-count-coll)
        prob-distribution ($= target-count-coll / s)]
    (gain-fn prob-distribution)))
    
(defn- ds-size [ds] (first (dim ds)))

(defn info-gain [ds target feature gain-fn]
  (let [entire (data-gain ds target gain-fn)
        grouped-map ($group-by feature ds)]
    ($= entire - 
        (sum (map (fn [sub-ds] (* (data-gain sub-ds target gain-fn)
                                  ($= (ds-size sub-ds) / (ds-size ds))))
                  (vals grouped-map))))))

(declare dtree make-node)

(defn dtree [ds target &
             {:keys [gain-fn] :or {gain-fn entropy}}]
  (cond (= 1 (count ($group-by target ds)))
        (if (sequential? ($ target ds))
          {:value (first ($ target ds))}
          {:value ($ target ds)})
        
        (= 1 (second (dim ds)))
        {:value (majority ($ target ds))}
        
        true
        (make-node ds target gain-fn)))

(defn- choose-feature [ds target gain-fn]
  (let [col-names (remove #(= % target) (col-names ds))
        gains (map #(info-gain ds target % gain-fn) col-names)
        gain-col-map (zipmap gains col-names)]
    (-> gain-col-map sort last val)))

(defn remove-feature [feature ds]
  (if (> (ncol ds) 2)
    ($ [:not feature] ds)
    (dataset (drop-while #(= feature %) (col-names ds))
             ($ [:not feature] ds))))

(defn- make-children [ds feature target gain-fn]
  (map (fn [[key sub-ds]]
         (if (= 1 (nrow sub-ds))
           [key (dtree sub-ds target :gain-fn gain-fn)]
           [key (dtree (remove-feature feature sub-ds)
                       target :gain-fn gain-fn)]))
       ($group-by feature ds)))

(defn- make-node [ds target gain-fn]
  (let [feature (choose-feature ds target gain-fn)
        children (make-children ds feature target gain-fn)]
    {:node feature :children children}))

(declare dtree-classify process-children)
(defn dtree-classify [tree data-point]
  (match [tree]
         [({:node n :children c} :only [:node :children])] (process-children c n data-point)
         [{:value v}] v))

(defn process-children [children node-name point]
  (let [node-val (node-name point)]
    (reduce
     (fn [prev child]
       (if (not (nil? prev)) prev
           (match [child]
                  [[{node-name node-val}
                    sub-tree]] (dtree-classify sub-tree point))))
     nil children)))

(defn dtree-classify-dataset [tree ds]
  (map (partial dtree-classify tree)
       (-> ds second second)))