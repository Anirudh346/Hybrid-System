import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';
import { ComparisonProvider } from './contexts/ComparisonContext';
import Navbar from './components/Navbar';
import ComparisonBar from './components/ComparisonBar';
import PrivateRoute from './components/PrivateRoute';
import Home from './pages/Home';
import Products from './pages/Products';
import ProductDetail from './pages/ProductDetail';
import AIRecommendation from './pages/AIRecommendation';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Comparison from './pages/Comparison';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <ComparisonProvider>
          <Router>
            <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
              <Navbar />
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/products" element={<Products />} />
                <Route path="/product/:id" element={<ProductDetail />} />
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />
                <Route path="/comparison" element={<Comparison />} />
                <Route
                  path="/ai-recommendation"
                  element={
                    <PrivateRoute>
                      <AIRecommendation />
                    </PrivateRoute>
                  }
                />
              </Routes>
              <ComparisonBar />
            </div>
          </Router>
        </ComparisonProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;

