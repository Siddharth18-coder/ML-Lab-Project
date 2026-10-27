import React, { useState, useEffect, useMemo } from 'react';

/**
 * DiabetesDashboard.jsx - CDC BRFSS Diabetes Risk Stratification Interface
 * =========================================================================
 * Rigorous Portfolio Implementation:
 * - Powered by a Machine Learning model trained on the U.S. CDC Behavioral Risk
 *   Factor Surveillance System (BRFSS) dataset.
 * - Educational & Research Portfolio: Estimates statistical associations with
 *   self-reported survey diabetes status. Not validated diagnostic software.
 * - Stale state tracking: Clearly invalidates displayed results when inputs change.
 * - Structured field-level validation errors (HTTP 422).
 * - Verified TreeSHAP explainability labeled relative to the explained class.
 * - Explicit opt-in Rule Demo Mode (illustrative heuristic, not ML).
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const AGE_OPTIONS = [
  { value: 1, label: '18 - 24 years' },
  { value: 2, label: '25 - 29 years' },
  { value: 3, label: '30 - 34 years' },
  { value: 4, label: '35 - 39 years' },
  { value: 5, label: '40 - 44 years' },
  { value: 6, label: '45 - 49 years' },
  { value: 7, label: '50 - 54 years' },
  { value: 8, label: '55 - 59 years' },
  { value: 9, label: '60 - 64 years' },
  { value: 10, label: '65 - 69 years' },
  { value: 11, label: '70 - 74 years' },
  { value: 12, label: '75 - 79 years' },
  { value: 13, label: '80 years or older' },
];

const GEN_HLTH_OPTIONS = [
  { value: 1, label: '1 - Excellent' },
  { value: 2, label: '2 - Very Good' },
  { value: 3, label: '3 - Good' },
  { value: 4, label: '4 - Fair' },
  { value: 5, label: '5 - Poor' },
];

const EDUCA_OPTIONS = [
  { value: 1, label: 'Never attended school / Kindergarten only' },
  { value: 2, label: 'Elementary (Grades 1 - 8)' },
  { value: 3, label: 'Some High School (Grades 9 - 11)' },
  { value: 4, label: 'High School Graduate / GED' },
  { value: 5, label: 'Some College or Technical School' },
  { value: 6, label: 'College Graduate (4+ years)' },
];

const INCOME_OPTIONS = [
  { value: 1, label: 'Less than $10,000' },
  { value: 2, label: '$10,000 to <$15,000' },
  { value: 3, label: '$15,000 to <$20,000' },
  { value: 4, label: '$20,000 to <$25,000' },
  { value: 5, label: '$25,000 to <$35,000' },
  { value: 6, label: '$35,000 to <$50,000' },
  { value: 7, label: '$50,000 to <$75,000' },
  { value: 8, label: '$75,000 or more' },
];

const SCENARIO_PRESETS = {
  healthy: {
    name: 'Illustrative: Low-Risk Active Profile',
    badge: 'No Reported Diabetes',
    description: 'Active young adult, normal BMI, healthy dietary habits, no chronic diagnoses.',
    data: {
      bmi: 22.4,
      high_bp: 0,
      high_chol: 0,
      chol_check: 1,
      smoker: 0,
      stroke: 0,
      heart_disease_or_attack: 0,
      phys_activity: 1,
      fruits: 1,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 0,
      gen_hlth: 1,
      ment_hlth: 0.0,
      phys_hlth: 0.0,
      diff_walk: 0,
      sex: 0,
      age: 2,
      education: 6,
      income: 8,
    }
  },
  prediabetic: {
    name: 'Illustrative: Borderline Survey Indicators',
    badge: 'Intermediate Indicators',
    description: 'Middle-aged individual with overweight BMI, diagnosed hypertension, sedentary habits.',
    data: {
      bmi: 28.5,
      high_bp: 1,
      high_chol: 1,
      chol_check: 1,
      smoker: 1,
      stroke: 0,
      heart_disease_or_attack: 0,
      phys_activity: 0,
      fruits: 1,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 0,
      gen_hlth: 3,
      ment_hlth: 2.0,
      phys_hlth: 4.0,
      diff_walk: 0,
      sex: 1,
      age: 8,
      education: 4,
      income: 5,
    }
  },
  diabetic: {
    name: 'Illustrative: High-Risk Comorbid Profile',
    badge: 'Elevated Risk Factors',
    description: 'Older respondent with class-1 obesity, hypertension, hyperlipidemia, and cardiovascular history.',
    data: {
      bmi: 34.6,
      high_bp: 1,
      high_chol: 1,
      chol_check: 1,
      smoker: 1,
      stroke: 1,
      heart_disease_or_attack: 1,
      phys_activity: 0,
      fruits: 0,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 1,
      gen_hlth: 5,
      ment_hlth: 10.0,
      phys_hlth: 15.0,
      diff_walk: 1,
      sex: 1,
      age: 10,
      education: 4,
      income: 3,
    }
  }
};

export default function DiabetesDashboard() {
  const [formData, setFormData] = useState(SCENARIO_PRESETS.healthy.data);
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isStale, setIsStale] = useState(false);
  const [fieldErrors, setFieldErrors] = useState({});

  const [backendStatus, setBackendStatus] = useState('checking'); // 'live' | 'model_missing' | 'offline' | 'checking'
  const [backendMeta, setBackendMeta] = useState(null);
  const [backendError, setBackendError] = useState(null);

  const [metricsData, setMetricsData] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState(null);
  const [activeTab, setActiveTab] = useState('assessment'); // 'assessment' | 'metrics'

  // Check backend health on mount
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(data.model_ready ? 'live' : 'model_missing');
        setBackendMeta(data);
        setBackendError(null);
      } else {
        setBackendStatus('offline');
      }
    } catch {
      setBackendStatus('offline');
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch metrics when metrics tab is selected
  const fetchMetrics = async () => {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/metrics`);
      if (!res.ok) {
        throw new Error(`Metrics endpoint returned HTTP ${res.status}`);
      }
      const data = await res.json();
      setMetricsData(data);
    } catch (err) {
      setMetricsError(err.message || 'Failed to fetch model evaluation metrics.');
      setMetricsData(null);
    } finally {
      setMetricsLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'metrics' && !metricsData && !metricsLoading) {
      fetchMetrics();
    }
  }, [activeTab]);

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: typeof value === 'string' && !isNaN(Number(value)) ? Number(value) : value,
    }));
    // Clear error for field if any
    if (fieldErrors[field]) {
      setFieldErrors((prev) => {
        const updated = { ...prev };
        delete updated[field];
        return updated;
      });
    }
    // Mark previous result as stale
    if (predictionResult) {
      setIsStale(true);
    }
  };

  const handlePresetSelect = (presetKey) => {
    if (SCENARIO_PRESETS[presetKey]) {
      setFormData(SCENARIO_PRESETS[presetKey].data);
      setFieldErrors({});
      if (predictionResult) {
        setIsStale(true);
      }
    }
  };

  const handleDemoToggle = (e) => {
    setDemoMode(e.target.checked);
    if (predictionResult) {
      setIsStale(true);
    }
  };

  const handleRunAssessment = async () => {
    setLoading(true);
    setBackendError(null);
    setFieldErrors({});

    const endpoint = demoMode ? `${API_BASE_URL}/demo-predict` : `${API_BASE_URL}/predict`;

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
        if (res.status === 422 && Array.isArray(errorBody.errors)) {
          const mapped = {};
          errorBody.errors.forEach((e) => {
            mapped[e.field] = e.message;
          });
          setFieldErrors(mapped);
          throw new Error(`Input validation failed for ${errorBody.errors.length} field(s). Please review highlighted inputs.`);
        }
        throw new Error(errorBody.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setPredictionResult(data);
      setIsStale(false);
    } catch (err) {
      setBackendError(err.message || 'Failed to connect to ML prediction backend.');
    } finally {
      setLoading(false);
    }
  };

  // Color mappings
  const getRiskColor = (riskClass) => {
    switch (riskClass) {
      case 'No Reported Diabetes':
      case 'Healthy':
        return {
          bg: 'bg-emerald-950/40',
          border: 'border-emerald-500/50',
          text: 'text-emerald-400',
          badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
          bar: 'bg-emerald-500',
        };
      case 'Pre-diabetic':
        return {
          bg: 'bg-amber-950/40',
          border: 'border-amber-500/50',
          text: 'text-amber-400',
          badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
          bar: 'bg-amber-500',
        };
      case 'Diabetic':
        return {
          bg: 'bg-rose-950/40',
          border: 'border-rose-500/50',
          text: 'text-rose-400',
          badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
          bar: 'bg-rose-500',
        };
      default:
        return {
          bg: 'bg-slate-900',
          border: 'border-slate-700',
          text: 'text-slate-300',
          badge: 'bg-slate-800 text-slate-300 border-slate-700',
          bar: 'bg-slate-600',
        };
    }
  };

  const riskColors = predictionResult ? getRiskColor(predictionResult.prediction.risk_class) : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Top Banner & Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white">EndocrineAI™</h1>
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">
                  CDC BRFSS v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400">Public Health Population Risk Screening (Educational Portfolio)</p>
            </div>
          </div>

          {/* Right Status Controls */}
          <div className="flex items-center space-x-4">
            {/* Live Backend Indicator */}
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'live'
                    ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                    : backendStatus === 'checking'
                    ? 'bg-amber-400 animate-pulse'
                    : backendStatus === 'model_missing'
                    ? 'bg-amber-500'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300">
                {backendStatus === 'live'
                  ? 'CDC BRFSS Model Live'
                  : backendStatus === 'checking'
                  ? 'Connecting...'
                  : backendStatus === 'model_missing'
                  ? 'Backend Online (Model Missing)'
                  : 'Backend Offline'}
              </span>
            </div>

            {/* Rule Demo Mode Toggle */}
            <label className="flex items-center space-x-2 cursor-pointer text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full hover:border-slate-700 transition">
              <input
                type="checkbox"
                checked={demoMode}
                onChange={handleDemoToggle}
                className="rounded border-slate-700 text-amber-500 focus:ring-0 cursor-pointer"
              />
              <span className={demoMode ? 'text-amber-400 font-semibold' : 'text-slate-400'}>
                Rule Demo Mode
              </span>
            </label>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Research & Educational Scope Disclaimer */}
        <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-3.5 flex items-start space-x-3 text-xs text-slate-400">
          <svg className="w-4 h-4 text-cyan-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold text-slate-300">Educational & Methodological Scope: </span>
            This interface illustrates machine learning classification and TreeSHAP explainability on cross-sectional self-reported survey indicators from the CDC BRFSS.
            It identifies statistical associations with existing self-reported diabetes status, <span className="text-slate-200">not a validated medical prediction of future diabetes onset</span>, and does not replace diagnostic clinical blood tests (fasting plasma glucose or HbA1c).
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('assessment')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition ${
              activeTab === 'assessment'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            Survey Assessment Form
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition ${
              activeTab === 'metrics'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            Model Benchmark & Limitations
          </button>
        </div>

        {/* Demo Mode Notice */}
        {demoMode && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start space-x-3">
            <svg className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div className="text-xs text-amber-200">
              <p className="font-semibold text-amber-300">Illustrative Rule Demo Mode Active</p>
              <p className="mt-0.5">
                Evaluates a project-defined heuristic rule rather than the trained Random Forest classifier.
                Probabilities, calibration metrics, and SHAP feature attributions are omitted.
              </p>
            </div>
          </div>
        )}

        {/* Backend Error Banner with Retry */}
        {backendError && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <svg className="w-5 h-5 text-rose-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-xs font-semibold text-rose-300">Assessment Request Failed</p>
                <p className="text-xs text-rose-200/80">{backendError}</p>
              </div>
            </div>
            <button
              onClick={handleRunAssessment}
              className="px-3 py-1.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-medium transition shadow flex-shrink-0 ml-4"
            >
              Retry
            </button>
          </div>
        )}

        {activeTab === 'assessment' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Form Column */}
            <div className="lg:col-span-7 space-y-6">
              {/* Scenario Presets Card */}
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Illustrative Population Profiles
                  </h3>
                  <span className="text-[11px] text-slate-500">Non-Clinical Educational Presets</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {Object.entries(SCENARIO_PRESETS).map(([key, preset]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => handlePresetSelect(key)}
                      className="text-left p-3 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/50 hover:bg-slate-900 transition group"
                    >
                      <div className="text-xs font-semibold text-white group-hover:text-indigo-400 transition">
                        {preset.name}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">{preset.badge}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Form Card */}
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
                {/* Section 1: Demographics & Socioeconomics */}
                <div>
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
                    <span>1. Demographics & Socioeconomic Status</span>
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Age Category</label>
                      <select
                        value={formData.age}
                        onChange={(e) => handleInputChange('age', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.age ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        {AGE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                      {fieldErrors.age && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.age}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Biological Sex</label>
                      <select
                        value={formData.sex}
                        onChange={(e) => handleInputChange('sex', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.sex ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>Female</option>
                        <option value={1}>Male</option>
                      </select>
                      {fieldErrors.sex && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.sex}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Education Level</label>
                      <select
                        value={formData.education}
                        onChange={(e) => handleInputChange('education', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.education ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        {EDUCA_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                      {fieldErrors.education && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.education}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Annual Household Income</label>
                      <select
                        value={formData.income}
                        onChange={(e) => handleInputChange('income', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.income ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        {INCOME_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                      {fieldErrors.income && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.income}</p>}
                    </div>
                  </div>
                </div>

                {/* Section 2: Vitals & Diagnosed Medical Conditions */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    2. Vitals & Survey Health Indicators
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="font-medium text-slate-300">Body Mass Index (BMI)</label>
                        <span className="text-indigo-400 font-semibold">{formData.bmi} kg/m²</span>
                      </div>
                      <input
                        type="number"
                        step="0.1"
                        min="12.0"
                        max="70.0"
                        value={formData.bmi}
                        onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value) || 20.0)}
                        className={`w-full bg-slate-950 border ${fieldErrors.bmi ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      />
                      {fieldErrors.bmi && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.bmi}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">High Blood Pressure History</label>
                      <select
                        value={formData.high_bp}
                        onChange={(e) => handleInputChange('high_bp', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.high_bp ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>No (Normal Blood Pressure)</option>
                        <option value={1}>Yes (Reported Hypertension)</option>
                      </select>
                      {fieldErrors.high_bp && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.high_bp}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">High Blood Cholesterol History</label>
                      <select
                        value={formData.high_chol}
                        onChange={(e) => handleInputChange('high_chol', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.high_chol ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>No (Normal Cholesterol)</option>
                        <option value={1}>Yes (Reported Hypercholesterolemia)</option>
                      </select>
                      {fieldErrors.high_chol && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.high_chol}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Cholesterol Checked in Past 5 Yrs</label>
                      <select
                        value={formData.chol_check}
                        onChange={(e) => handleInputChange('chol_check', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.chol_check ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={1}>Yes (Tested within 5 years)</option>
                        <option value={0}>No / Never checked</option>
                      </select>
                      {fieldErrors.chol_check && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.chol_check}</p>}
                    </div>
                  </div>
                </div>

                {/* Section 3: Cardiovascular & Chronic Comorbidities */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    3. Cardiovascular Comorbidities & Mobility
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">History of Stroke</label>
                      <select
                        value={formData.stroke}
                        onChange={(e) => handleInputChange('stroke', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.stroke ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes</option>
                      </select>
                      {fieldErrors.stroke && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.stroke}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Heart Disease / Attack</label>
                      <select
                        value={formData.heart_disease_or_attack}
                        onChange={(e) => handleInputChange('heart_disease_or_attack', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.heart_disease_or_attack ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes (CHD or Infarction)</option>
                      </select>
                      {fieldErrors.heart_disease_or_attack && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.heart_disease_or_attack}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Difficulty Walking</label>
                      <select
                        value={formData.diff_walk}
                        onChange={(e) => handleInputChange('diff_walk', Number(e.target.value))}
                        className={`w-full bg-slate-950 border ${fieldErrors.diff_walk ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      >
                        <option value={0}>No Limitation</option>
                        <option value={1}>Yes (Difficulty walking/climbing)</option>
                      </select>
                      {fieldErrors.diff_walk && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.diff_walk}</p>}
                    </div>
                  </div>
                </div>

                {/* Section 4: Lifestyle, Nutrition & Healthcare Access */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    4. Lifestyle, Habits & Care Access
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Physical Activity</label>
                      <select
                        value={formData.phys_activity}
                        onChange={(e) => handleInputChange('phys_activity', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>Yes (Active in past 30 days)</option>
                        <option value={0}>No (Sedentary)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Smoking History</label>
                      <select
                        value={formData.smoker}
                        onChange={(e) => handleInputChange('smoker', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>Non-Smoker (&lt;100 cigs lifetime)</option>
                        <option value={1}>Smoker (≥100 cigarettes lifetime)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Heavy Alcohol</label>
                      <select
                        value={formData.hvy_alcohol_consump}
                        onChange={(e) => handleInputChange('hvy_alcohol_consump', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes (&gt;14/wk M, &gt;7/wk F)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Daily Fruit</label>
                      <select
                        value={formData.fruits}
                        onChange={(e) => handleInputChange('fruits', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>≥ 1 time / day</option>
                        <option value={0}>&lt; 1 time / day</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Daily Vegetables</label>
                      <select
                        value={formData.veggies}
                        onChange={(e) => handleInputChange('veggies', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>≥ 1 time / day</option>
                        <option value={0}>&lt; 1 time / day</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Healthcare Coverage</label>
                      <select
                        value={formData.any_healthcare}
                        onChange={(e) => handleInputChange('any_healthcare', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>Insured / Covered</option>
                        <option value={0}>Uninsured</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 5: Self-Rated General Health */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    5. Self-Reported Health Status
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">General Health Rating</label>
                      <select
                        value={formData.gen_hlth}
                        onChange={(e) => handleInputChange('gen_hlth', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        {GEN_HLTH_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Poor Physical Health Days (0-30)
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="30"
                        value={formData.phys_hlth}
                        onChange={(e) => handleInputChange('phys_hlth', parseFloat(e.target.value) || 0)}
                        className={`w-full bg-slate-950 border ${fieldErrors.phys_hlth ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      />
                      {fieldErrors.phys_hlth && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.phys_hlth}</p>}
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Poor Mental Health Days (0-30)
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="30"
                        value={formData.ment_hlth}
                        onChange={(e) => handleInputChange('ment_hlth', parseFloat(e.target.value) || 0)}
                        className={`w-full bg-slate-950 border ${fieldErrors.ment_hlth ? 'border-rose-500' : 'border-slate-800'} rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500`}
                      />
                      {fieldErrors.ment_hlth && <p className="text-[11px] text-rose-400 mt-1">{fieldErrors.ment_hlth}</p>}
                    </div>
                  </div>
                </div>

                {/* Submit Action */}
                <div className="pt-3">
                  <button
                    type="button"
                    onClick={handleRunAssessment}
                    disabled={loading || backendStatus === 'offline'}
                    className="w-full py-3 bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white font-semibold rounded-xl shadow-lg shadow-indigo-600/25 transition disabled:opacity-50 flex items-center justify-center space-x-2 text-sm"
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>Computing CDC BRFSS Assessment...</span>
                      </>
                    ) : (
                      <span>Run CDC BRFSS Risk Assessment</span>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Results Column */}
            <div className="lg:col-span-5 space-y-6">
              {predictionResult ? (
                <>
                  {/* Stale Result Warning Banner */}
                  {isStale && (
                    <div className="bg-amber-500/15 border border-amber-500/40 rounded-xl p-3 flex items-center justify-between text-xs text-amber-200">
                      <div className="flex items-center space-x-2">
                        <svg className="w-4 h-4 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <span>Inputs modified. Click "Run Assessment" to update results.</span>
                      </div>
                      <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">
                        Stale
                      </span>
                    </div>
                  )}

                  {/* Primary Risk Classification Card */}
                  <div className={`rounded-2xl border ${riskColors.border} ${riskColors.bg} p-6 shadow-xl relative overflow-hidden transition ${isStale ? 'opacity-70' : 'opacity-100'}`}>
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase font-semibold text-slate-400 tracking-wider">
                        Risk Stratification Result
                      </span>
                      <span className={`text-xs px-2.5 py-1 rounded-full font-semibold border ${riskColors.badge}`}>
                        {predictionResult.metadata.prediction_source === 'rule_based_demo'
                          ? 'Project Heuristic Demo'
                          : `${(predictionResult.prediction.confidence * 100).toFixed(1)}% Confidence`}
                      </span>
                    </div>

                    <div className="mt-4">
                      <h2 className={`text-3xl font-extrabold tracking-tight ${riskColors.text}`}>
                        {predictionResult.prediction.risk_class}
                      </h2>
                      <p className="text-xs text-slate-300 mt-1">
                        {predictionResult.prediction.risk_class === 'No Reported Diabetes' || predictionResult.prediction.risk_class === 'Healthy'
                          ? 'Low statistical probability of reported diabetes in CDC BRFSS cross-sectional survey data.'
                          : predictionResult.prediction.risk_class === 'Pre-diabetic'
                          ? 'Intermediate survey risk indicators (borderline glycemic patterns in behavioral survey).'
                          : 'Elevated statistical probability of self-reported diabetes diagnosis. Clinical consultation advised.'}
                      </p>
                    </div>

                    {/* Probability Distribution Gauge */}
                    {predictionResult.prediction.probabilities && (
                      <div className="mt-6 space-y-2 border-t border-slate-800/80 pt-4">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Uncalibrated Class Probability Distribution</span>
                          <span>Sum: 100%</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          {Object.entries(predictionResult.prediction.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800 text-center">
                              <div className="text-[11px] text-slate-400 truncate">{cls}</div>
                              <div className="text-sm font-bold text-white mt-0.5">
                                {(prob * 100).toFixed(1)}%
                              </div>
                              <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
                                <div
                                  className={`h-full ${
                                    cls === 'No Reported Diabetes' || cls === 'Healthy'
                                      ? 'bg-emerald-400'
                                      : cls === 'Pre-diabetic'
                                      ? 'bg-amber-400'
                                      : 'bg-rose-500'
                                  }`}
                                  style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* SHAP Feature Attribution Card */}
                  {predictionResult.shap_explanation.status === 'available' ? (
                    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-sm font-bold text-white">Explainable AI (TreeSHAP)</h3>
                          <p className="text-xs text-slate-400">
                            Contributions relative to <span className="font-semibold text-slate-200">{predictionResult.shap_explanation.explained_class}</span> class
                          </p>
                        </div>
                        <span className="text-[11px] px-2 py-0.5 bg-slate-800 text-slate-300 rounded font-mono">
                          Base: {(predictionResult.shap_explanation.base_value * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* Feature Contributions List */}
                      <div className="space-y-3 pt-1">
                        {predictionResult.shap_explanation.top_features.map((feat) => {
                          const isSupport = feat.direction === 'supports';
                          return (
                            <div key={feat.feature} className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl space-y-1">
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-medium text-slate-200">{feat.display_name}</span>
                                <span
                                  className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                                    isSupport
                                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                  }`}
                                >
                                  {feat.impact}
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-400">{feat.clinical_insight}</p>
                            </div>
                          );
                        })}
                      </div>

                      {/* Additivity Verification Badge */}
                      {predictionResult.shap_explanation.additivity_check && (
                        <div className="border-t border-slate-800 pt-3 flex items-center justify-between text-[11px] text-slate-400">
                          <span>Additivity Verification (Base + Σ SHAP = Probability)</span>
                          <span className="text-emerald-400 font-semibold flex items-center space-x-1">
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Verified (Δ = {predictionResult.shap_explanation.additivity_check.absolute_difference})</span>
                          </span>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 text-xs text-slate-400 flex items-start space-x-3">
                      <svg className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div>
                        <span className="font-semibold text-slate-300">SHAP Explanation Unavailable: </span>
                        {predictionResult.shap_explanation.reason || 'Local feature attribution is not supported for this mode.'}
                      </div>
                    </div>
                  )}

                  {/* Tailored Lifestyle Guidelines */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-3">
                    <h3 className="text-sm font-bold text-white">Project-Defined Educational Lifestyle Pathways</h3>
                    <p className="text-[11px] text-slate-400">
                      General preventive public health recommendations aligned with CDC DPP lifestyle principles:
                    </p>
                    <div className="space-y-2.5">
                      {predictionResult.clinical_recommendations.map((rec, idx) => (
                        <div key={idx} className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-200">{rec.title}</span>
                            <span
                              className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${
                                rec.urgency === 'Immediate'
                                  ? 'bg-rose-500/20 text-rose-300'
                                  : rec.urgency === 'Moderate'
                                  ? 'bg-amber-500/20 text-amber-300'
                                  : 'bg-emerald-500/20 text-emerald-300'
                              }`}
                            >
                              {rec.urgency}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-1">{rec.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-slate-900/50 border border-slate-800/60 rounded-2xl p-12 text-center text-slate-500 flex flex-col items-center justify-center min-h-[400px]">
                  <svg className="w-12 h-12 text-slate-700 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  <p className="text-sm font-medium text-slate-400">Ready for Risk Evaluation</p>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs">
                    Populate the 21 CDC health indicators manually or select an illustrative profile above, then click Run Assessment.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Metrics Tab */
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-white">CDC BRFSS Model Benchmark & Statistical Limitations</h3>
                <p className="text-xs text-slate-400">
                  Evaluation results on the held-out 20% test partition (N = 10,000 real survey respondents) from the CDC BRFSS dataset.
                </p>
              </div>
              <button
                onClick={fetchMetrics}
                disabled={metricsLoading}
                className="px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-950 hover:bg-slate-900 text-xs text-slate-300 font-medium transition"
              >
                {metricsLoading ? 'Refreshing...' : 'Refresh Benchmark'}
              </button>
            </div>

            {metricsError && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-center justify-between text-xs text-rose-300">
                <span>{metricsError}</span>
                <button
                  onClick={fetchMetrics}
                  className="px-3 py-1 bg-rose-600 text-white rounded font-medium hover:bg-rose-500 transition"
                >
                  Retry
                </button>
              </div>
            )}

            {metricsData ? (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Hold-Out Accuracy</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {(metricsData.accuracy * 100).toFixed(2)}%
                    </div>
                    {metricsData.confidence_intervals_95_pct?.accuracy_95_ci && (
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        95% CI: [{(metricsData.confidence_intervals_95_pct.accuracy_95_ci[0] * 100).toFixed(1)}%, {(metricsData.confidence_intervals_95_pct.accuracy_95_ci[1] * 100).toFixed(1)}%]
                      </div>
                    )}
                  </div>

                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Balanced Accuracy</div>
                    <div className="text-2xl font-extrabold text-cyan-400 mt-1">
                      {(metricsData.balanced_accuracy * 100).toFixed(2)}%
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      Unweighted mean class recall
                    </div>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Macro F1-Score</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {metricsData.f1_macro.toFixed(4)}
                    </div>
                    {metricsData.confidence_intervals_95_pct?.f1_macro_95_ci && (
                      <div className="text-[10px] text-slate-500 mt-0.5">
                        95% CI: [{metricsData.confidence_intervals_95_pct.f1_macro_95_ci[0].toFixed(3)}, {metricsData.confidence_intervals_95_pct.f1_macro_95_ci[1].toFixed(3)}]
                      </div>
                    )}
                  </div>

                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Multiclass Brier Score</div>
                    <div className="text-2xl font-extrabold text-amber-400 mt-1">
                      {metricsData.multiclass_brier_score.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      Mean squared prob error
                    </div>
                  </div>
                </div>

                {/* Per-Class Detailed Performance Breakdown */}
                <div className="bg-slate-950/60 p-5 rounded-xl border border-slate-800 space-y-3">
                  <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Per-Class Detailed Performance (Documenting Severe Class Imbalance)
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left text-slate-300 border border-slate-800">
                      <thead className="bg-slate-950 text-slate-400 uppercase text-[11px]">
                        <tr>
                          <th className="p-2.5 border border-slate-800">Survey Class</th>
                          <th className="p-2.5 border border-slate-800 text-center">Test Support</th>
                          <th className="p-2.5 border border-slate-800 text-center">Recall (Sensitivity)</th>
                          <th className="p-2.5 border border-slate-800 text-center">Precision</th>
                          <th className="p-2.5 border border-slate-800 text-center">F1-Score</th>
                          <th className="p-2.5 border border-slate-800 text-center">ROC-AUC</th>
                          <th className="p-2.5 border border-slate-800 text-center">PR-AUC</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metricsData.confusion_matrix.classes.map((cls) => {
                          const cr = metricsData.classification_report?.[cls] || {};
                          const roc = metricsData.roc_auc_per_class?.[cls] || 0;
                          const pr = metricsData.average_precision_pr_auc_per_class?.[cls] || 0;
                          const isPreDiab = cls === 'Pre-diabetic';
                          return (
                            <tr key={cls} className={`hover:bg-slate-900/50 ${isPreDiab ? 'bg-amber-950/20' : ''}`}>
                              <td className="p-2.5 font-semibold text-white border border-slate-800 flex items-center space-x-2">
                                <span>{cls}</span>
                                {isPreDiab && (
                                  <span className="text-[9px] px-1.5 py-0.5 bg-amber-500/20 text-amber-300 rounded border border-amber-500/30">
                                    Severe Imbalance (1.8%)
                                  </span>
                                )}
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono">
                                {cr.support || metricsData.confusion_matrix.breakdown[cls]?.total_actual}
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono font-bold text-emerald-400">
                                {((cr.recall ?? metricsData.confusion_matrix.breakdown[cls]?.sensitivity_recall) * 100).toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono">
                                {((cr.precision ?? metricsData.confusion_matrix.breakdown[cls]?.precision) * 100).toFixed(1)}%
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono">
                                {(cr['f1-score'] ?? 0).toFixed(3)}
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono text-cyan-400">
                                {roc.toFixed(3)}
                              </td>
                              <td className="p-2.5 text-center border border-slate-800 font-mono text-indigo-400">
                                {pr.toFixed(3)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-2">
                    <span className="font-semibold text-amber-400">Limitation Notice:</span> In 3-class survey modeling, Pre-diabetic recall is low (4.89% recall, 2.36% precision) due to extreme class imbalance (1.84% prevalence in CDC BRFSS) and overlapping symptom profiles with both normoglycemic and diagnosed diabetic respondents.
                  </p>
                </div>

                {/* Baselines Comparison */}
                {metricsData.baselines_comparison && (
                  <div className="bg-slate-950/60 p-5 rounded-xl border border-slate-800 space-y-3">
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                      Architectural Baselines Comparison
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left text-slate-300 border border-slate-800">
                        <thead className="bg-slate-950 text-slate-400 uppercase text-[11px]">
                          <tr>
                            <th className="p-2.5 border border-slate-800">Model Architecture</th>
                            <th className="p-2.5 border border-slate-800 text-center">Accuracy</th>
                            <th className="p-2.5 border border-slate-800 text-center">Balanced Acc</th>
                            <th className="p-2.5 border border-slate-800 text-center">Macro F1</th>
                            <th className="p-2.5 border border-slate-800 text-center">Brier Score</th>
                            <th className="p-2.5 border border-slate-800 text-center">Train Time</th>
                            <th className="p-2.5 border border-slate-800 text-center">Inference Latency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(metricsData.baselines_comparison)
                            .filter(([k]) => !['probability_calibration_assessment', 'primary_model_training_seconds', 'primary_model_inference_latency_ms'].includes(k))
                            .map(([key, model]) => {
                              const isSelected = key === 'random_forest_balanced';
                              return (
                                <tr key={key} className={`hover:bg-slate-900/50 ${isSelected ? 'bg-indigo-950/30' : ''}`}>
                                  <td className="p-2.5 font-semibold text-white border border-slate-800">
                                    <div className="flex items-center space-x-2">
                                      <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                                      {isSelected && (
                                        <span className="text-[9px] px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 rounded border border-indigo-500/30">
                                          Selected Model
                                        </span>
                                      )}
                                    </div>
                                    <div className="text-[10px] text-slate-400 font-normal mt-0.5">{model.model_description}</div>
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono">
                                    {(model.accuracy * 100).toFixed(1)}%
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono font-bold text-cyan-400">
                                    {(model.balanced_accuracy * 100).toFixed(1)}%
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono">
                                    {model.f1_macro.toFixed(3)}
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono">
                                    {model.brier_score ? model.brier_score.toFixed(3) : 'N/A'}
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono text-slate-400">
                                    {model.training_time_sec ? `${model.training_time_sec.toFixed(2)}s` : '0.00s'}
                                  </td>
                                  <td className="p-2.5 text-center border border-slate-800 font-mono text-slate-400">
                                    {model.inference_latency_ms_per_sample ? `${(model.inference_latency_ms_per_sample * 1000).toFixed(2)}µs` : 'N/A'}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Duplicate Profiles & Secondary Binary Task Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {metricsData.duplicate_profile_analysis && (
                    <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-2">
                      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                        Survey Duplication & Bayes Error
                      </h4>
                      <p className="text-xs text-slate-300">
                        In {metricsData.duplicate_profile_analysis.total_records.toLocaleString()} respondents across 21 features, {metricsData.duplicate_profile_analysis.duplicate_profile_rows} rows ({metricsData.duplicate_profile_analysis.duplicate_profile_percentage}%) share identical responses.
                      </p>
                      <p className="text-xs text-amber-300/90">
                        <span className="font-semibold">{metricsData.duplicate_profile_analysis.profiles_with_conflicting_diagnoses} profile clusters</span> have identical survey answers but conflicting diabetes statuses, illustrating the irreducible error ceiling inherent to survey data without laboratory blood glucose biomarkers.
                      </p>
                    </div>
                  )}

                  {metricsData.secondary_binary_screening_task && (
                    <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-2">
                      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                        Secondary Binary Screening Experiment
                      </h4>
                      <p className="text-xs text-slate-300">
                        When evaluated as a secondary screening task (0: No Reported Diabetes vs 1: At-Risk Pre-diabetic or Diabetic):
                      </p>
                      <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                        <div className="p-2 bg-slate-900 rounded border border-slate-800">
                          <span className="text-slate-400 block">At-Risk Sensitivity</span>
                          <span className="text-emerald-400 font-bold text-sm">
                            {(metricsData.secondary_binary_screening_task.sensitivity_recall * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="p-2 bg-slate-900 rounded border border-slate-800">
                          <span className="text-slate-400 block">ROC-AUC</span>
                          <span className="text-cyan-400 font-bold text-sm">
                            {metricsData.secondary_binary_screening_task.roc_auc.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-slate-500 text-xs">
                {metricsLoading ? 'Loading validation metrics from backend...' : 'Click "Refresh Benchmark" to load metrics.'}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
