X, y = data_assignment3.get_iris_data()

feature_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

# Plot setup
plt.figure(figsize=(15, 10))

for i, (f1, f2) in enumerate(feature_pairs):
    plt.subplot(2, 3, i + 1)
    plt.scatter(X[:, f1], X[:, f2], c=y, cmap='viridis', s=15)  
    plt.title(f'Features {f1+1} & {f2+1}')
    plt.grid(True)

    # Optional: tighten axis limits slightly
    padding_x = (X[:, f1].max() - X[:, f1].min()) * 0.05
    padding_y = (X[:, f2].max() - X[:, f2].min()) * 0.05
    plt.xlim(X[:, f1].min() - padding_x, X[:, f1].max() + padding_x)
    plt.ylim(X[:, f2].min() - padding_y, X[:, f2].max() + padding_y)

plt.suptitle('Iris Data: Different Feature Pair Projections ', fontsize=14)
plt.tight_layout()
plt.show()


# Feature pairs
X_bad = X[:, [0, 1]]   
X_best = X[:, [2, 3]]  

# PCA to 2D
_, X_pca = pca(X, m=2)


plt.figure(figsize=(18, 5))

# Weak feature pair
plt.subplot(1, 3, 1)
plt.scatter(X_bad[:, 0], X_bad[:, 1], c=y, cmap='viridis', s=20)
plt.xlabel('(F1)')
plt.ylabel('(F2)')
plt.title('Weak Feature Pair (F1 & F2)')
plt.grid(True)
plt.axis('equal')

# Strong feature pair
plt.subplot(1, 3, 2)
plt.scatter(X_best[:, 0], X_best[:, 1], c=y, cmap='viridis', s=20)
plt.xlabel('(F3)')
plt.ylabel('(F4)')
plt.title('Best Feature Pair (F3 & F4)')
plt.grid(True)
plt.axis('equal')

# 3. PCA projection
plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=20)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection (2D)')
plt.grid(True)
plt.axis('equal')

plt.suptitle('Iris Data: Feature Pair Comparison vs PCA', fontsize=14)
plt.tight_layout()
plt.show()

