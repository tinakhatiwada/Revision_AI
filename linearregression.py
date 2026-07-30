import numpy as np
W_true=[3, -2, 5]
b_true = 4
rng=np.random.default_rng(seed=42)
X=rng.random(size=(13,3))
y = np.dot(X,np.array(W_true)) + b_true +rng.normal(loc=0.0,scale=0.1,size=(13,))

def compute_cost(X, y, w,b):
    m = X.shape[0]
    predictions = X @ w + b
    J=(1/(2*m))*np.sum(np.power((predictions-y),2))
    return J

def compute_gradient(X,y,w,b):
    m=X.shape[0]
    predictions=X @ w + b
    djw=(1/m)* X.T @ (predictions-y)
    djb=(1/m)*np.sum((predictions-y))
    return djw,djb

def gradient_descent(X,y,w,b,alpha,minimum_iterations):
    i=0
    previous_cost=0
    while True:
        dw,db=compute_gradient(X,y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
        i+=1
        if i%20 == 0 :
            print(f"Iteration {i}: cost = {compute_cost(X, y, w, b)}")
        current_cost=compute_cost(X,y,w,b)
        if abs(previous_cost - current_cost) < 1e-8:
            print("The cost is saturated")
            break
        previous_cost=current_cost
        if i>=minimum_iterations:
            print("the minimum_iterations have exceeded")
            break
        if np.linalg.norm(dw) < 1e-6 or abs(db) < 1e-6:
            print("the gardients are approaching 0")
            break
    return w,b

wi=np.zeros(3)
bi=0
w_final, b_final = gradient_descent(X, y, wi, bi, alpha=0.1, minimum_iterations=2000)
print("Learned w:", w_final)
print("Learned b:", b_final)
print("True w:", W_true, "True b:", b_true)











