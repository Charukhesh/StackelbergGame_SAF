import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3D)
from scipy.optimize import minimize
import time

def emissions_aviation_fuel():
    return 9.5  # kg CO2 per gallon

def emissions_saf():
    return 3.78  # kg CO2 per gallon

def revenue(alpha):
    return 9.7 + 3.0 * np.tanh(1.5 * alpha)  # billion dollars
    #return 9.7

def operating_cost():
    return 5.66  # billion dollars without fuel

def fuel_qty():
    return 1.495  # billion gallons (fixed baseline per user)


def aviation_fuel_price(qty):
    p0 = 2.10
    q0 = 1.495
    # elasticity = 0.1  # <-- BUG: Exponent is 10
    elasticity = 2.0  # <-- FIX: Exponent is 0.5, price is stable
    q = max(qty, 1e-9)
    raw = p0 * (q / q0)**(1.0 / elasticity)
    # price floor to avoid pathological zeros in extreme cases
    return max(raw, 1.5)

def saf_price(qty):
    p0 = 6.3          # <-- FIX: High base price
    q0 = 0.75          # <-- FIX: For a small initial quantity
    elasticity = 1.2  # <-- FIX: Price rises as demand grows

    q = max(qty, 1e-9)
    raw = p0 * (q / q0)**(1.0 / elasticity)
    return max(raw, 1.50)

def compute_profit_for_alpha(alpha, subsidy_per_gal, carbon_tax_per_ton):
    """
    alpha: SAF blend fraction in [0,1] (fraction of total fuel that is SAF)
    subsidy_per_gal: $/gal subsidy paid for SAF
    carbon_tax_per_ton: $/ton CO2
    Returns profit in billion $ and breakdown dict
    """
    # quantities in billion gallons
    q_total_b = fuel_qty()
    q_saf_b = q_total_b * alpha
    q_conv_b = q_total_b * (1.0 - alpha)

    # unit prices ($/gal)
    p_conv = aviation_fuel_price(q_conv_b)
    p_saf = saf_price(q_saf_b)

    # fuel cost in billion $
    # fuel cost in billion $
    fuel_cost_b = (p_conv * q_conv_b) + (p_saf * q_saf_b)

    # carbon tax: carbon_tax_billion = tau ($/ton) * emissions_kg_per_gal * qty_billion / 1000
    tau = carbon_tax_per_ton
    carbon_tax_conv_b = tau * emissions_aviation_fuel() * q_conv_b / 1000.0
    carbon_tax_saf_b = tau * emissions_saf() * q_saf_b / 1000.0
    carbon_tax_b = carbon_tax_conv_b + carbon_tax_saf_b

    # subsidy income in billion $ (applies only to SAF gallons)
    subsidy_income_b = (subsidy_per_gal * q_saf_b)

    # profit (billion $)
    profit_b = revenue(alpha) - operating_cost() - fuel_cost_b - carbon_tax_b + subsidy_income_b

    return profit_b, {
        'profit_billion': profit_b,
        'fuel_cost_billion': fuel_cost_b,
        'carbon_tax_billion': carbon_tax_b,
        'subsidy_income_billion': subsidy_income_b,
        'p_conv_per_gal': p_conv,
        'p_saf_per_gal': p_saf,
        'qty_conv_billion': q_conv_b,
        'qty_saf_billion': q_saf_b
    }
alpha = np.linspace(0,0.99,100)
profit = []

for i in range (0,len(alpha)):
  profit_b, info = compute_profit_for_alpha(alpha[i],subsidy_per_gal=4.5, carbon_tax_per_ton=90)
  profit.append(profit_b)

profit = np.array(profit)
max_index = np.argmax(profit)
alpha_max = alpha[max_index]
profit_max = profit[max_index]

print(f"Maximum profit = {profit_max:.3f} billion $ at alpha = {alpha_max:.3f}")

plt.plot(alpha, profit, label='Profit vs Alpha')
plt.scatter(alpha_max, profit_max, color='red', label=f'Max Profit at α={alpha_max:.2f}')
plt.title('Profit vs Alpha')
plt.xlabel('Alpha (SAF fraction)')
plt.ylabel('Profit (billion $)')
plt.legend()
plt.grid(True)
plt.show()

def objective_func_to_minimize(alpha, subsidy, carbon_tax):
    """
    Wrapper function for scipy.optimize.minimize.
    We want to MAXIMIZE profit, so we MINIMIZE -profit.
    """
    # Ensure alpha is a single value, not an array
    alpha_val = alpha[0]
    
    # This calls your main compute_profit_for_alpha function
    profit, info = compute_profit_for_alpha(alpha_val, 
                                            subsidy_per_gal=subsidy, 
                                            carbon_tax_per_ton=carbon_tax)
    
    # Return negative profit
    return -profit

def create_optimal_alpha_contour_map():
    """
    Runs the optimization for a grid of subsidy and carbon tax values
    and plots the resulting optimal alpha as a contour map.
    """
    
    # --- Set up the Grid ---
    n_points = 30
    subsidy_range = np.linspace(0, 5.0, n_points)      # $0 to $5.0/gal
    carbon_tax_range = np.linspace(0, 200, n_points)   # $0 to $200/ton

    # Create a grid to store the results
    optimal_alpha_grid = np.zeros((n_points, n_points))

    # --- Run the Optimization Loop ---
    print("\nRunning 2D optimization grid... (This may take a moment)")
    start_time = time.time()

    # Initial guess for alpha
    initial_guess = [0.5]
    # Bounds for alpha (must be between 0 and 1)
    bnds = [(0.0, 1.0)]

    # Loop over tax (y-axis)
    for i, tax in enumerate(carbon_tax_range):
        # Loop over subsidy (x-axis)
        for j, subsidy in enumerate(subsidy_range):
            
            # Run the minimization to find the best alpha
            res = minimize(objective_func_to_minimize, 
                           initial_guess, 
                           args=(subsidy, tax), 
                           method='L-BFGS-B',
                           bounds=bnds)
            
            # Store the optimal alpha
            optimal_alpha_grid[i, j] = res.x[0]

    end_time = time.time()
    print(f"Optimization complete in {end_time - start_time:.2f} seconds.")

    # --- Plotting the Heatmap ---
    plt.figure(figsize=(10, 8))

    # Use extent to label the axes correctly
    # [x_min, x_max, y_min, y_max]
    extent = [subsidy_range[0], subsidy_range[-1], carbon_tax_range[0], carbon_tax_range[-1]]

    # Plot the heatmap
    # origin='lower' puts (0,0) at the bottom left
    img = plt.imshow(optimal_alpha_grid, extent=extent, origin='lower', aspect='auto', cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(img)
    cbar.set_label('Optimal Alpha (SAF fraction)', fontsize=12)

    # Add contour lines for clarity
    contours = plt.contour(subsidy_range, carbon_tax_range, optimal_alpha_grid, 
                           levels=8, colors='white', alpha=0.9) # <-- Increased alpha for brighter lines
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    plt.xlabel('Subsidy ($/gal)', fontsize=12)
    plt.ylabel('Carbon Tax ($/ton)', fontsize=12)
    plt.title('Optimal SAF Alpha vs. Policy Levers (Subsidy & Carbon Tax)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

# --- Run the 2D contour plot function ---
create_optimal_alpha_contour_map()  