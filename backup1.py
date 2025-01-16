from scipy.optimize import least_squares
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
from ttkbootstrap.dialogs import Messagebox
import pandas as pd
from tkinter import filedialog
import clipboard

class DataFittingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Fitting Tool")
        self.root.geometry("1200x800")
        
        self.style = ttk.Style()
        self.theme_var = ttk.StringVar(value="light")
        self.rounding_enabled = ttk.BooleanVar(value=True)
        self.degree_var = ttk.StringVar(value="Best Fit")
        self.fit_type = ttk.StringVar(value="Polynomial")
        self.r_squared = ttk.StringVar(value="")
        
        # Available models
        self.models = {
            "Polynomial": lambda x, params: sum(params[i] * x**i for i in range(len(params))),
            "Exponential": lambda x, params: params[0] * np.exp(params[1] * x),
            "Power Law": lambda x, params: params[0] * x**params[1],
            "Logarithmic": lambda x, params: params[0] + params[1] * np.log(x + 1e-10),
            "Sinusoidal": lambda x, params: params[0] * np.sin(params[1] * x + params[2]) + params[3],
            "Gaussian": lambda x, params: params[0] * np.exp(-(x - params[1])**2 / (2 * params[2]**2))
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container with padding
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=BOTH, expand=YES)

        # Create two columns
        left_frame = ttk.Frame(container)
        left_frame.pack(side=LEFT, fill=BOTH, expand=NO, padx=(0, 20))
        
        right_frame = ttk.Frame(container)
        right_frame.pack(side=LEFT, fill=BOTH, expand=YES)
        
        # Data Input Section
        data_frame = ttk.LabelFrame(left_frame, text="Data Input", padding=10)
        data_frame.pack(fill=X, pady=(0, 15))
        
        # Data input buttons
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=X, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Paste from Excel",
            command=self.paste_from_clipboard,
            style='info.TButton',
            width=15
        ).pack(side=LEFT, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Load CSV",
            command=self.load_csv,
            style='info.TButton',
            width=15
        ).pack(side=LEFT, padx=2)
        
        ttk.Button(
            btn_frame,
            text="Clear Data",
            command=self.clear_data,
            style='danger.TButton',
            width=15
        ).pack(side=LEFT, padx=2)
        
        # Manual data entry
        ttk.Label(data_frame, text="X Data Points:").pack(anchor=W, pady=(10, 5))
        self.entry_x = ttk.Text(data_frame, width=40, height=3)
        self.entry_x.pack(pady=(0, 5))
        self.entry_x.insert('1.0', "-2, -1, 0, 1, 2")
        
        ttk.Label(data_frame, text="Y Data Points:").pack(anchor=W, pady=(0, 5))
        self.entry_y = ttk.Text(data_frame, width=40, height=3)
        self.entry_y.pack(pady=(0, 5))
        self.entry_y.insert('1.0', "0.1, 0.5, 1, 2, 4")
        
        # Settings Section
        settings_frame = ttk.LabelFrame(left_frame, text="Settings", padding=10)
        settings_frame.pack(fill=X, pady=(0, 15))
        
        # Theme selector
        ttk.Label(settings_frame, text="Theme:").pack(anchor=W)
        theme_menu = ttk.OptionMenu(
            settings_frame, 
            self.theme_var,
            "light",
            *["light", "dark"],
            command=self.change_theme
        )
        theme_menu.pack(anchor=W, pady=(0, 10))
        
        # Options
        ttk.Checkbutton(
            settings_frame, 
            text="Enable Rounding", 
            variable=self.rounding_enabled,
            style='primary.TCheckbutton'
        ).pack(anchor=W, pady=(0, 10))
        
        # Model Selection Section
        model_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding=10)
        model_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Label(model_frame, text="Fitting Type:").pack(anchor=W)
        self.fit_menu = ttk.Combobox(
            model_frame, 
            textvariable=self.fit_type,
            values=list(self.models.keys()),
            state="readonly"
        )
        self.fit_menu.pack(anchor=W, pady=(0, 10))
        self.fit_menu.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Polynomial degree selector (initially hidden)
        self.degree_frame = ttk.Frame(model_frame)
        ttk.Label(self.degree_frame, text="Polynomial Degree:").pack(anchor=W)
        self.degree_menu = ttk.Combobox(
            self.degree_frame,
            textvariable=self.degree_var,
            values=["Best Fit"] + [str(i) for i in range(1, 10)],
            state="readonly"
        )
        self.degree_menu.pack(anchor=W, pady=(0, 10))
        
        # Fit button
        ttk.Button(
            left_frame,
            text="Fit Model",
            command=self.fit_model,
            style='success.TButton',
            width=20
        ).pack(anchor=W, pady=(0, 15))
        
        # Results Section
        results_frame = ttk.LabelFrame(left_frame, text="Results", padding=10)
        results_frame.pack(fill=X)
        
        # R-squared value
        self.r_squared_label = ttk.Label(
            results_frame,
            textvariable=self.r_squared,
            font=("Helvetica", 10)
        )
        self.r_squared_label.pack(anchor=W, pady=(0, 5))
        
        # Model equation
        self.label_result = ttk.Label(
            results_frame,
            wraplength=400,
            font=("Helvetica", 10)
        )
        self.label_result.pack(anchor=W)
        
        # Right frame for plot
        self.frame_right = right_frame
        
        # Initial UI update
        self.on_model_change()
        
    def on_model_change(self, event=None):
        if self.fit_type.get() == "Polynomial":
            self.degree_frame.pack(fill=X)
        else:
            self.degree_frame.pack_forget()
            
    def paste_from_clipboard(self):
        try:
            # Get clipboard content
            data = clipboard.paste()
            # Split into rows
            rows = [row.strip() for row in data.split('\n') if row.strip()]
            
            if len(rows) < 2:
                raise ValueError("Not enough data in clipboard")
                
            # Convert to numpy arrays
            data = np.array([row.split('\t') for row in rows], dtype=float)
            
            # Update entry fields
            self.entry_x.delete('1.0', END)
            self.entry_y.delete('1.0', END)
            self.entry_x.insert('1.0', ', '.join(map(str, data[:, 0])))
            self.entry_y.insert('1.0', ', '.join(map(str, data[:, 1])))
            
            Messagebox.show_info("Success", "Data pasted successfully!")
        except Exception as e:
            Messagebox.show_error("Error", f"Failed to paste data: {str(e)}")
            
    def load_csv(self):
        try:
            filename = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                df = pd.read_csv(filename)
                if len(df.columns) < 2:
                    raise ValueError("CSV must have at least 2 columns")
                
                self.entry_x.delete('1.0', END)
                self.entry_y.delete('1.0', END)
                self.entry_x.insert('1.0', ', '.join(map(str, df.iloc[:, 0])))
                self.entry_y.insert('1.0', ', '.join(map(str, df.iloc[:, 1])))
                
                Messagebox.show_info("Success", "Data loaded successfully!")
        except Exception as e:
            Messagebox.show_error("Error", f"Failed to load CSV: {str(e)}")
            
    def clear_data(self):
        self.entry_x.delete('1.0', END)
        self.entry_y.delete('1.0', END)
        self.label_result.config(text="")
        self.r_squared.set("")
        # Clear plot
        for widget in self.frame_right.winfo_children():
            widget.destroy()

    def parse_numbers(self, text):
        """Parse numbers from text, properly handling negative numbers"""
        text = text.get('1.0', END)  # Get all text from Text widget
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return np.array([float(n) for n in numbers])

    def change_theme(self, _=None):
        theme = 'darkly' if self.theme_var.get() == 'dark' else 'litera'
        self.style.theme_use(theme)
        plt.style.use('dark_background' if self.theme_var.get() == 'dark' else 'default')
        if hasattr(self, 'last_plot_data'):
            self.update_plot(*self.last_plot_data)

    def calculate_r_squared(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_initial_guess(self, model_type):
        if model_type == "Polynomial":
            degree = int(self.degree_var.get()) if self.degree_var.get() != "Best Fit" else 1
            return np.ones(degree + 1)
        elif model_type == "Exponential":
            return [1, 0.1]
        elif model_type == "Power Law":
            return [1, 1]
        elif model_type == "Logarithmic":
            return [1, 1]
        elif model_type == "Sinusoidal":
            return [1, 1, 0, 0]
        elif model_type == "Gaussian":
            return [1, 0, 1]
        return [1, 1]  # Default

    def format_equation(self, model_type, params):
        if model_type == "Polynomial":
            terms = []
            for i, coef in enumerate(params):
                coef = self.smart_round(coef)
                if abs(coef) > 1e-6:
                    if i == 0:
                        terms.append(f"{coef:g}")
                    elif i == 1:
                        terms.append(f"{coef:g}x")
                    else:
                        terms.append(f"{coef:g}x^{i}")
            return " + ".join(terms).replace("+ -", "- ")
        elif model_type == "Exponential":
            return f"{self.smart_round(params[0])} * exp({self.smart_round(params[1])} * x)"
        elif model_type == "Power Law":
            return f"{self.smart_round(params[0])} * x^{self.smart_round(params[1])}"
        elif model_type == "Logarithmic":
            return f"{self.smart_round(params[0])} + {self.smart_round(params[1])} * ln(x)"
        elif model_type == "Sinusoidal":
            return f"{self.smart_round(params[0])} * sin({self.smart_round(params[1])}x + {self.smart_round(params[2])}) + {self.smart_round(params[3])}"
        elif model_type == "Gaussian":
            return f"{self.smart_round(params[0])} * exp(-(x - {self.smart_round(params[1])})^2 / (2 * {self.smart_round(params[2])}^2))"
        return str(params)

    def smart_round(self, value):
        if not self.rounding_enabled.get():
            return value
        rounded = round(value, 2)
        if abs(rounded - round(value)) < 0.01:
            return round(value)
        elif abs(rounded - round(value, 1)) < 0.01:
            return round(value, 1)
        return rounded

    def residuals(self, params, x, y, model_type):
        model_func = self.models[model_type]
        return y - model_func(x, params)

    def fit_model(self):
        try:
            x_data = self.parse_numbers(self.entry_x)
            y_data = self.parse_numbers(self.entry_y)
            
            if len(x_data) != len(y_data):
                raise ValueError("X and Y data must have the same number of points")
            
            if len(x_data) < 2:
                raise ValueError("At least two data points are required")
            
            fit_type = self.fit_type.get()
            if fit_type == "Polynomial":
                best_r2 = float('-inf')
                best_params = None
                best_degree = None
                
                # Determine best polynomial degree if "Best Fit" is selected
                max_degree = min(9, len(x_data) - 1)  # Avoid overfitting for small datasets
                degrees_to_try = (
                    range(1, max_degree + 1)
                    if self.degree_var.get() == "Best Fit"
                    else [int(self.degree_var.get())]
                )
                
                for degree in degrees_to_try:
                    initial_guess = np.ones(degree + 1)
                    result = least_squares(self.residuals, initial_guess, args=(x_data, y_data, "Polynomial"))
                    params = result.x
                    y_pred = self.models["Polynomial"](x_data, params)
                    r2 = self.calculate_r_squared(y_data, y_pred)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = params
                        best_degree = degree
                
                # Use best results
                best_params = best_params
                self.r_squared.set(f"R² = {best_r2:.4f}")
                model_desc = self.format_equation("Polynomial", best_params)
                self.label_result.config(text=f"Fitted Model (Degree {best_degree}):\n{model_desc}", font=30)
            else:
                # Handle non-polynomial models
                initial_guess = self.get_initial_guess(fit_type)
                result = least_squares(self.residuals, initial_guess, args=(x_data, y_data, fit_type))
                best_params = result.x
                y_pred = self.models[fit_type](x_data, best_params)
                r2 = self.calculate_r_squared(y_data, y_pred)
                self.r_squared.set(f"R² = {r2:.4f}")
                model_desc = self.format_equation(fit_type, best_params)
                self.label_result.config(text=f"Fitted Model:\n{model_desc}", font=30)
            
            self.update_plot(x_data, y_data, best_params, fit_type)
    
        except Exception as e:
            Messagebox.show_error("Error", str(e))
            self.label_result.config(text="")
            self.r_squared.set("")

    def update_plot(self, x_data, y_data, coefficients, fit_type):
        # Store plot data for theme changes
        self.last_plot_data = (x_data, y_data, coefficients, fit_type)
        
        # Clear previous plot
        for widget in self.frame_right.winfo_children():
            widget.destroy()
        
        # Create new plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # Set style based on theme
        plt.style.use('dark_background' if self.theme_var.get() == 'dark' else 'default')
        
        # Generate smooth curve for plotting
        x_range = max(x_data) - min(x_data)
        x_fit = np.linspace(min(x_data) - x_range*0.1, max(x_data) + x_range*0.1, 500)
        
        try:
            y_fit = self.models[fit_type](x_fit, coefficients)
            
            # Plot data and fitted curve
            scatter_color = '#00bc8c' if self.theme_var.get() == 'dark' else '#007bff'
            line_color = '#f39c12' if self.theme_var.get() == 'dark' else '#dc3545'
            
            # Plot data points
            ax.scatter(x_data, y_data, color=scatter_color, label='Data Points', zorder=5)
            
            # Plot fitted curve
            ax.plot(x_fit, y_fit, color=line_color, label='Fitted Model', linewidth=2, zorder=4)
            
            # Add grid with lower opacity
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            
            # Customize plot
            ax.legend(fontsize=10)
            ax.set_title("Data Points and Fitted Model", fontsize=12, pad=10)
            ax.set_xlabel("X", fontsize=10, labelpad=8)
            ax.set_ylabel("Y", fontsize=10, labelpad=8)
            
            # Add minor ticks
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', alpha=0.2)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas and pack it
            canvas = FigureCanvasTkAgg(fig, master=self.frame_right)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
            
        except Exception as e:
            Messagebox.show_error("Plot Error", f"Failed to create plot: {str(e)}")

def main():
    root = ttk.Window(themename="litera")
    app = DataFittingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()