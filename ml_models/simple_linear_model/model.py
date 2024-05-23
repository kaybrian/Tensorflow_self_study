#  define the data you need to  train from 
data = {
    'size': [650, 785, 1200, 1400, 1800],
    'price': [77250, 92850, 150000, 178000, 215000]
}

# Build a function to scale the data 
def normalize(values):
    min_val = min(values)
    max_value = max(values)
    return [(x - min_val) / (max_value - min_val) for x in values]


# Normalize the values in the data 
data['size'] = normalize(data['size'])
data['price'] = normalize(data['price'])


# now lets begin by assigning the data to bias and weights 
m = 0
b = 0
learning_rate = 0.01
epochs = 1000


# create the cost function of our model 
def cost(m, b, data):
    total_cost = 0
    N = len(data['size'])
    
    for i in range(N):
        x = data['size'][i]
        y = data['price'][i]
        total_cost += (y - (m * x + b)) ** 2
    
    return total_cost / N 

# create the gradient descent function
def gradient_descent(m, b, data, learning_rate):
    N = len(data['size'])
    m_gradient = 0
    b_gradient = 0
    
    for i in range(N):
        x = data['size'][i]
        y = data['price'][i]
        m_gradient += (-2/N) * x * (y - (m * x + b))
        b_gradient += (-2/N) * (y - (m * x + b))
        
    m = m - learning_rate * m_gradient
    b = b - learning_rate * b_gradient
    
    return m, b

# Train the data to see the values at work 
for epoch in range(epochs):
    m,b = gradient_descent(m, b, data, learning_rate)
    if epoch % 100 == 0:
        print("Epoch: {}, Cost: {}".format(epoch, cost(m, b, data)))
        print(f" Weight(M): {m}, Bias(b): {b}")
        


# Make predictions 
def predict(size, m, b):
    return m * size + b
   
# Predicting price for a normalized house size (for demonstration purposes)
normalized_size = (1000 - 650) / (1800 - 650)  # Assuming a house size of 1000 sq ft
predicted_price = predict(m, b, normalized_size)
print(f"Predicted normalized price for house size of 1000 sq ft: {predicted_price}")     