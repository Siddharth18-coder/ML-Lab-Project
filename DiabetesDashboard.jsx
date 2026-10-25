import React, { useState, useEffect, useMemo } from 'react';

/**
 * DiabetesDashboard.jsx - Educational Clinical Risk Intelligence Interface
 * =========================================================================
 * Scope & Transparency:
 * - Built as an educational prototype for classifying current diabetes risk tiers
 *   (Healthy, Pre-diabetic, Diabetic) based on synthetic benchmark data.
 * - Connects to the FastAPI backend with configurable API_BASE_URL.
 * - Does NOT silently fall back to simulation when backend errors occur.
 * - Offline simulation is strictly an explicitly selected Demo Mode.
 * - Features truthful SHAP explainability labeled relative to the explained class.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

// -----------------------------------------------------------------------------
// Clinical Reference Boundaries & Physiological Limits
// -----------------------------------------------------------------------------
const CLINICAL_FIELDS = {
  // Demographics
  age: { label: 'Age', unit: 'years', min: 18, max: 100, normal: '18 - 65', step: 1, category: 'demographics' },
  gender: { label: 'Gender', type: 'select', options: ['Female', 'Male', 'Other'], category: 'demographics' },
  ethnicity: { label: 'Ethnicity / Race', type: 'select', options: ['Caucasian', 'African American', 'Asian', 'Hispanic', 'Other'], category: 'demographics' },
  family_history_diabetes: { label: 'Family History (T2D)', type: 'select', options: [{ label: 'No (Negative)', value: 0 }, { label: 'Yes (Immediate Family)', value: 1 }], category: 'demographics' },

  // Anthropometrics & Blood Pressure
  bmi: { label: 'Body Mass Index (BMI)', unit: 'kg/m²', min: 12.0, max: 70.0, normal: '18.5 - 24.9', step: 0.1, category: 'vitals' },
  systolic_bp: { label: 'Systolic Blood Pressure', unit: 'mmHg', min: 60, max: 260, normal: '< 120', step: 1, category: 'vitals' },
  diastolic_bp: { label: 'Diastolic Blood Pressure', unit: 'mmHg', min: 40, max: 160, normal: '< 80', step: 1, category: 'vitals' },

  // Laboratory Panel (Optional: KNNImputer will impute if omitted)
  fasting_blood_glucose: { label: 'Fasting Plasma Glucose', unit: 'mg/dL', min: 40, max: 450, normal: '70 - 99', step: 1, category: 'labs' },
  hba1c: { label: 'Glycated Hemoglobin (HbA1c)', unit: '%', min: 3.5, max: 18.0, normal: '< 5.7%', step: 0.1, category: 'labs' },
  total_cholesterol: { label: 'Total Serum Cholesterol', unit: 'mg/dL', min: 80, max: 450, normal: '< 200', step: 1, category: 'labs' },
  hdl_cholesterol: { label: 'HDL Cholesterol', unit: 'mg/dL', min: 15, max: 140, normal: '≥ 50', step: 1, category: 'labs' },
  ldl_cholesterol: { label: 'LDL Cholesterol', unit: 'mg/dL', min: 30, max: 320, normal: '< 100', step: 1, category: 'labs' },
  triglycerides: { label: 'Serum Triglycerides', unit: 'mg/dL', min: 30, max: 750, normal: '< 150', step: 1, category: 'labs' },
  serum_creatinine: { label: 'Serum Creatinine', unit: 'mg/dL', min: 0.2, max: 8.0, normal: '0.6 - 1.2', step: 0.05, category: 'labs' },

  // Lifestyle & Daily Habits
  physical_activity_minutes_per_week: { label: 'Physical Activity', unit: 'min/wk', min: 0, max: 1200, normal: '≥ 150', step: 10, category: 'lifestyle' },
  smoking_status: { label: 'Smoking Status', type: 'select', options: ['Never', 'Former', 'Current'], category: 'lifestyle' },
  alcohol_intake: { label: 'Alcohol Intake', type: 'select', options: ['None', 'Moderate', 'Heavy'], category: 'lifestyle' },
  diet_quality_score: { label: 'Diet Quality Score', unit: 'scale 1-10', min: 1.0, max: 10.0, normal: '≥ 7.0', step: 0.5, category: 'lifestyle' },
  sleep_hours: { label: 'Nightly Sleep', unit: 'hours/day', min: 2.0, max: 16.0, normal: '7.0 - 9.0', step: 0.5, category: 'lifestyle' },
};

// -----------------------------------------------------------------------------
// Preconfigured Synthetic Scenario Profiles for Demonstration
// -----------------------------------------------------------------------------
const SCENARIO_PRESETS = {
  healthy: {
    name: 'Healthy Scenario',
    badge: 'Normoglycemic Baseline',
    data: {
      age: 32.0,
      gender: 'Female',
      ethnicity: 'Caucasian',
      family_history_diabetes: 0,
      bmi: 21.8,
      systolic_bp: 114.0,
      diastolic_bp: 72.0,
      fasting_blood_glucose: 88.0,
      hba1c: 5.1,
      total_cholesterol: 170.0,
      hdl_cholesterol: 62.0,
      ldl_cholesterol: 92.0,
      triglycerides: 80.0,
      serum_creatinine: 0.8,
      physical_activity_minutes_per_week: 220.0,
      smoking_status: 'Never',
      alcohol_intake: 'None',
      diet_quality_score: 8.5,
      sleep_hours: 7.5,
    }
  },
  prediabetic: {
    name: 'Pre-diabetic Scenario',
    badge: 'Impaired Glycemic Pattern',
    data: {
      age: 51.0,
      gender: 'Male',
      ethnicity: 'Hispanic',
      family_history_diabetes: 1,
      bmi: 29.4,
      systolic_bp: 134.0,
      diastolic_bp: 84.0,
      fasting_blood_glucose: 114.0,
      hba1c: 6.1,
      total_cholesterol: 216.0,
      hdl_cholesterol: 41.0,
      ldl_cholesterol: 138.0,
      triglycerides: 190.0,
      serum_creatinine: 1.05,
      physical_activity_minutes_per_week: 60.0,
      smoking_status: 'Former',
      alcohol_intake: 'Moderate',
      diet_quality_score: 5.0,
      sleep_hours: 6.2,
    }
  },
  diabetic: {
    name: 'Diabetic Scenario',
    badge: 'Hyperglycemic Pattern',
    data: {
      age: 58.0,
      gender: 'Male',
      ethnicity: 'African American',
      family_history_diabetes: 1,
      bmi: 34.6,
      systolic_bp: 148.0,
      diastolic_bp: 92.0,
      fasting_blood_glucose: 168.0,
      hba1c: 8.6,
      total_cholesterol: 246.0,
      hdl_cholesterol: 33.0,
      ldl_cholesterol: 156.0,
      triglycerides: 275.0,
      serum_creatinine: 1.35,
      physical_activity_minutes_per_week: 20.0,
      smoking_status: 'Current',
      alcohol_intake: 'Heavy',
      diet_quality_score: 3.0,
      sleep_hours: 5.5,
    }
  }
};

// -----------------------------------------------------------------------------
// SVG Icons
// -----------------------------------------------------------------------------
const Icons = {
  Activity: () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  ShieldCheck: () => (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
  AlertTriangle: () => (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  ),
  AlertOctagon: () => (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
    </svg>
  ),
  HeartPulse: () => (
    <svg className="w-5 h-5 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
    </svg>
  ),
  BarChart3: () => (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  ),
  Zap: () => (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
};

export default function DiabetesDashboard() {
  const [formData, setFormData] = useState(SCENARIO_PRESETS.prediabetic.data);
  const [activeTab, setActiveTab] = useState('assessment'); // 'assessment' | 'metrics'
  const [activeFormSection, setActiveFormSection] = useState('labs');
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [validationErrors, setValidationErrors] = useState({});
  const [apiError, setApiError] = useState(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  // Health and Metrics State
  const [apiHealth, setApiHealth] = useState({ status: 'checking', model_ready: false });
  const [metricsData, setMetricsData] = useState(null);
  const [metricsError, setMetricsError] = useState(null);

  // Check backend health on mount and when tab changes to metrics
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiHealth({ status: 'connected', model_ready: Boolean(data.model_ready) });
      } else {
        setApiHealth({ status: 'server_error', model_ready: false });
      }
    } catch {
      setApiHealth({ status: 'offline', model_ready: false });
    }
  };

  const fetchMetrics = async () => {
    try {
      setMetricsError(null);
      const res = await fetch(`${API_BASE_URL}/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetricsData(data);
      } else {
        const err = await res.json().catch(() => ({ detail: 'Failed to fetch metrics.' }));
        setMetricsError(err.detail || 'Metrics unavailable from backend.');
        setMetricsData(null);
      }
    } catch (e) {
      setMetricsError('Cannot connect to API server to fetch evaluation metrics.');
      setMetricsData(null);
    }
  };

  useEffect(() => {
    checkBackendHealth();
  }, []);

  useEffect(() => {
    if (activeTab === 'metrics') {
      fetchMetrics();
    }
  }, [activeTab]);

  // Exact validation matching backend Pydantic models
  const validateForm = (data) => {
    const errors = {};
    Object.keys(CLINICAL_FIELDS).forEach((key) => {
      const field = CLINICAL_FIELDS[key];
      const val = data[key];
      if (val !== null && val !== undefined && val !== '') {
        if (field.min !== undefined && val < field.min) {
          errors[key] = `Value cannot be below clinical minimum (${field.min} ${field.unit || ''})`;
        } else if (field.max !== undefined && val > field.max) {
          errors[key] = `Value cannot exceed clinical maximum (${field.max} ${field.unit || ''})`;
        }
      }
    });

    // Exact blood pressure boundary rule: Systolic must be at least 15 mmHg higher than Diastolic
    if (data.systolic_bp < data.diastolic_bp + 15) {
      errors.systolic_bp = `Systolic BP (${data.systolic_bp} mmHg) must be at least 15 mmHg higher than Diastolic BP (${data.diastolic_bp} mmHg).`;
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleInputChange = (field, value) => {
    const updated = {
      ...formData,
      [field]: value === '' ? null : (typeof formData[field] === 'number' ? parseFloat(value) : value)
    };
    setFormData(updated);
    validateForm(updated);
  };

  const loadPreset = (presetKey) => {
    const preset = SCENARIO_PRESETS[presetKey];
    if (preset) {
      setFormData(preset.data);
      setValidationErrors({});
      setApiError(null);
    }
  };

  // Run assessment (Trained Model or Explicit Demo)
  const handleRunAssessment = async (e) => {
    if (e) e.preventDefault();
    setApiError(null);

    if (!validateForm(formData)) {
      return;
    }

    setIsLoading(true);

    const endpoint = isDemoMode ? `${API_BASE_URL}/demo-predict` : `${API_BASE_URL}/predict`;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const result = await response.json();
        setPredictionResult(result);
        setIsLoading(false);
      } else {
        const errData = await response.json().catch(() => ({ detail: 'Unknown error occurred.' }));
        let errMessage = errData.detail || 'Request failed';
        if (response.status === 422) {
          errMessage = `Validation Error: ${JSON.stringify(errData.detail)}`;
        } else if (response.status === 503) {
          errMessage = `Model Unavailable: ${errData.detail || 'Model artifacts are not loaded on backend.'}`;
        }
        setApiError({ status: response.status, message: errMessage });
        setPredictionResult(null);
        setIsLoading(false);
      }
    } catch (err) {
      setApiError({
        status: 'network_error',
        message: `Network Connection Error: Could not reach backend service at ${API_BASE_URL}. Ensure uvicorn is running.`
      });
      setPredictionResult(null);
      setIsLoading(false);
    }
  };

  // Dynamic Tier Styling
  const tierConfig = useMemo(() => {
    if (!predictionResult) return null;
    const tier = predictionResult.prediction.risk_class;
    if (tier === 'Diabetic') {
      return {
        tier: 'Diabetic',
        badge: 'Diabetic Pattern Identified',
        badgeBg: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
        accent: 'rose',
        icon: Icons.AlertOctagon,
      };
    } else if (tier === 'Pre-diabetic') {
      return {
        tier: 'Pre-diabetic',
        badge: 'Impaired Glycemic Pattern',
        badgeBg: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
        accent: 'amber',
        icon: Icons.AlertTriangle,
      };
    } else {
      return {
        tier: 'Healthy',
        badge: 'Normoglycemic Pattern',
        badgeBg: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
        accent: 'emerald',
        icon: Icons.ShieldCheck,
      };
    }
  }, [predictionResult]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Navigation Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-cyan-500 rounded-xl shadow-lg shadow-indigo-500/20">
              <Icons.Activity />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white">EndocrineAI™ Risk Classifier</h1>
                <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                  Educational Prototype
                </span>
              </div>
              <p className="text-xs text-slate-400">Classifying Current Glycemic Categories with SHAP Feature Explanations</p>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs">
            {/* Backend Status Indicator */}
            <div className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800">
              <span
                className={`w-2 h-2 rounded-full ${
                  apiHealth.status === 'connected' && apiHealth.model_ready
                    ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                    : apiHealth.status === 'connected'
                    ? 'bg-amber-400'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300">
                {apiHealth.status === 'connected' && apiHealth.model_ready
                  ? 'FastAPI Model Live'
                  : apiHealth.status === 'connected'
                  ? 'FastAPI Live (Model Unloaded)'
                  : 'Backend Offline'}
              </span>
            </div>

            {/* Navigation Tabs */}
            <div className="flex rounded-lg bg-slate-900 p-0.5 border border-slate-800">
              <button
                onClick={() => setActiveTab('assessment')}
                className={`px-3 py-1.5 rounded-md font-medium transition ${
                  activeTab === 'assessment' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Patient Assessment
              </button>
              <button
                onClick={() => setActiveTab('metrics')}
                className={`px-3 py-1.5 rounded-md font-medium transition ${
                  activeTab === 'metrics' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Model Validation Metrics
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Synthetic Benchmark Disclaimer Banner */}
        <div className="mb-6 bg-slate-950/60 border border-indigo-500/20 rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2 text-xs text-slate-300">
            <span className="px-2 py-0.5 rounded font-bold uppercase tracking-wider bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Educational Notice
            </span>
            <span>
              This model was trained on <strong>synthetic benchmark data</strong>. High accuracy reflects synthetic generator rules and is not evidence of real-world clinical performance.
            </span>
          </div>
          <div className="flex items-center space-x-2 text-xs">
            <label className="flex items-center space-x-2 cursor-pointer bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800 hover:border-slate-700">
              <input
                type="checkbox"
                checked={isDemoMode}
                onChange={(e) => {
                  setIsDemoMode(e.target.checked);
                  setApiError(null);
                }}
                className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="text-slate-300 font-medium">Explicit Rule Demo Mode</span>
            </label>
          </div>
        </div>

        {/* Demo Mode Notice */}
        {isDemoMode && (
          <div className="mb-6 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Icons.AlertTriangle />
              <span>
                <strong>Active Mode: Rule-based demo — not a trained model prediction.</strong> Confidence probabilities and SHAP values are disabled in demo mode.
              </span>
            </div>
            <button
              onClick={() => setIsDemoMode(false)}
              className="text-amber-400 underline hover:text-amber-300 font-semibold"
            >
              Switch to Trained ML Model
            </button>
          </div>
        )}

        {/* Preset Scenarios */}
        <div className="mb-6 bg-slate-950/40 border border-slate-800/80 rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <Icons.Zap />
            <span className="font-medium text-slate-300">Load Synthetic Benchmark Profile:</span>
          </div>
          <div className="flex flex-wrap gap-2.5">
            <button
              type="button"
              onClick={() => loadPreset('healthy')}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/20 transition"
            >
              Healthy Scenario
            </button>
            <button
              type="button"
              onClick={() => loadPreset('prediabetic')}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 transition"
            >
              Pre-diabetic Scenario
            </button>
            <button
              type="button"
              onClick={() => loadPreset('diabetic')}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 transition"
            >
              Diabetic Scenario
            </button>
          </div>
        </div>

        {activeTab === 'assessment' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Input Form Column */}
            <div className="lg:col-span-5 bg-slate-950/80 border border-slate-800 rounded-3xl p-6 shadow-xl backdrop-blur">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <h2 className="text-base font-semibold text-white">Patient Profile Inputs</h2>
                  <p className="text-xs text-slate-400">Enter laboratory panel, vital signs, and demographics</p>
                </div>
                <button
                  type="button"
                  onClick={handleRunAssessment}
                  disabled={isLoading}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-xs rounded-xl shadow-lg shadow-indigo-600/30 transition flex items-center space-x-1.5 disabled:opacity-50"
                >
                  {isLoading ? <span>Evaluating...</span> : <span>Run Assessment</span>}
                </button>
              </div>

              {/* Form Category Navigation */}
              <div className="grid grid-cols-4 gap-1 p-1 bg-slate-900 rounded-xl mb-5 text-xs">
                {[
                  { id: 'labs', label: 'Labs' },
                  { id: 'vitals', label: 'Vitals' },
                  { id: 'demographics', label: 'Demographics' },
                  { id: 'lifestyle', label: 'Lifestyle' },
                ].map((sec) => (
                  <button
                    key={sec.id}
                    type="button"
                    onClick={() => setActiveFormSection(sec.id)}
                    className={`py-1.5 font-medium rounded-lg transition ${
                      activeFormSection === sec.id ? 'bg-slate-800 text-indigo-400 shadow' : 'text-slate-400 hover:text-slate-300'
                    }`}
                  >
                    {sec.label}
                  </button>
                ))}
              </div>

              {/* Form Fields */}
              <form onSubmit={handleRunAssessment} className="space-y-4">
                {Object.keys(CLINICAL_FIELDS)
                  .filter((key) => CLINICAL_FIELDS[key].category === activeFormSection)
                  .map((key) => {
                    const field = CLINICAL_FIELDS[key];
                    const val = formData[key] ?? '';
                    const error = validationErrors[key];

                    return (
                      <div key={key} className="space-y-1">
                        <div className="flex justify-between items-center text-xs">
                          <label htmlFor={key} className="font-medium text-slate-300">
                            {field.label}
                          </label>
                          <span className="text-[11px] text-slate-500">
                            Normal: <span className="text-slate-400">{field.normal || 'Standard'}</span>
                          </span>
                        </div>

                        {field.type === 'select' ? (
                          <select
                            id={key}
                            value={val}
                            onChange={(e) => handleInputChange(key, e.target.value)}
                            className="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                          >
                            {field.options.map((opt) => {
                              const isObj = typeof opt === 'object';
                              const optVal = isObj ? opt.value : opt;
                              const optLabel = isObj ? opt.label : opt;
                              return (
                                <option key={optVal} value={optVal}>
                                  {optLabel}
                                </option>
                              );
                            })}
                          </select>
                        ) : (
                          <div className="relative">
                            <input
                              id={key}
                              type="number"
                              step={field.step || 1}
                              min={field.min}
                              max={field.max}
                              value={val}
                              placeholder="Optional (KNN imputed if blank)"
                              onChange={(e) => handleInputChange(key, e.target.value)}
                              className={`w-full bg-slate-900/90 border rounded-xl px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 transition ${
                                error ? 'border-rose-500/80 focus:ring-rose-500/50' : 'border-slate-700/80 focus:ring-indigo-500/50'
                              }`}
                            />
                            {field.unit && (
                              <span className="absolute right-3 top-2.5 text-xs font-medium text-slate-500 pointer-events-none">
                                {field.unit}
                              </span>
                            )}
                          </div>
                        )}

                        {error && (
                          <p className="text-[11px] text-rose-400 mt-1 flex items-center space-x-1">
                            <span>⚠</span>
                            <span>{error}</span>
                          </p>
                        )}
                      </div>
                    );
                  })}

                {activeFormSection === 'vitals' && (
                  <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 text-[11px] text-slate-400">
                    💡 Clinical Validation Rule: Systolic BP must be at least 15 mmHg higher than Diastolic BP.
                  </div>
                )}
              </form>
            </div>

            {/* Results Column */}
            <div className="lg:col-span-7 space-y-6">
              {/* Error Display */}
              {apiError && (
                <div className="bg-rose-500/10 border border-rose-500/30 rounded-3xl p-6 text-rose-200 space-y-3">
                  <div className="flex items-center space-x-2 font-bold text-sm text-rose-300">
                    <Icons.AlertOctagon />
                    <span>Assessment Failed ({apiError.status})</span>
                  </div>
                  <p className="text-xs leading-relaxed text-rose-200">{apiError.message}</p>
                  <button
                    type="button"
                    onClick={handleRunAssessment}
                    className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-xs font-semibold transition"
                  >
                    Retry Assessment
                  </button>
                </div>
              )}

              {/* Assessment Results Card */}
              {predictionResult ? (
                <>
                  <div className="bg-slate-950/90 border border-slate-800 rounded-3xl p-6 shadow-2xl backdrop-blur space-y-5">
                    {/* Header with Source Badge */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                            Classified Glycemic Tier
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                              predictionResult.metadata.prediction_source === 'trained_model'
                                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                                : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                            }`}
                          >
                            Source: {predictionResult.metadata.prediction_source === 'trained_model' ? 'Trained Model' : 'Rule-Based Demo'}
                          </span>
                        </div>

                        <div className="flex items-center space-x-3 mt-1.5">
                          <h2 className="text-3xl font-extrabold text-white tracking-tight">
                            {predictionResult.prediction.risk_class}
                          </h2>
                          {tierConfig && (
                            <span className={`px-3 py-1 rounded-full text-xs font-bold border ${tierConfig.badgeBg}`}>
                              {tierConfig.badge}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Confidence / Mode Box */}
                      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-3 text-right min-w-[130px]">
                        <span className="text-[10px] text-slate-400 block font-medium">Model Output</span>
                        <span className="text-xl font-black text-white">
                          {predictionResult.prediction.confidence !== null
                            ? `${(predictionResult.prediction.confidence * 100).toFixed(1)}%`
                            : 'N/A (Demo)'}
                        </span>
                      </div>
                    </div>

                    {/* Probability Distribution */}
                    {predictionResult.prediction.probabilities ? (
                      <div className="space-y-2">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-medium text-slate-300">Softmax Class Probabilities</span>
                          <span className="text-[11px] text-slate-500">Multinomial Model Output</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-center text-xs">
                          {[
                            { label: 'Healthy', val: predictionResult.prediction.probabilities.Healthy, color: 'bg-emerald-500', text: 'text-emerald-400' },
                            { label: 'Pre-diabetic', val: predictionResult.prediction.probabilities['Pre-diabetic'], color: 'bg-amber-500', text: 'text-amber-400' },
                            { label: 'Diabetic', val: predictionResult.prediction.probabilities.Diabetic, color: 'bg-rose-500', text: 'text-rose-400' },
                          ].map((item) => (
                            <div key={item.label} className="bg-slate-900/90 border border-slate-800/80 rounded-xl p-2.5">
                              <span className="text-[11px] text-slate-400 block">{item.label}</span>
                              <span className={`text-base font-bold ${item.text} block my-1`}>
                                {(item.val * 100).toFixed(1)}%
                              </span>
                              <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className={`${item.color} h-full rounded-full transition-all duration-500`}
                                  style={{ width: `${Math.min(100, Math.max(4, item.val * 100))}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="p-3 bg-slate-900/40 rounded-xl border border-slate-800 text-xs text-slate-400">
                        Probabilities are not computed in rule-based demo mode.
                      </div>
                    )}
                  </div>

                  {/* SHAP Feature Explanations */}
                  <div className="bg-slate-950/90 border border-slate-800 rounded-3xl p-6 shadow-xl backdrop-blur space-y-4">
                    <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                      <div>
                        <div className="flex items-center space-x-2">
                          <Icons.BarChart3 />
                          <h3 className="text-base font-semibold text-white">SHAP Feature Attribution</h3>
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Explaining prediction for: <strong>{predictionResult.shap_explanation.explained_class || predictionResult.prediction.risk_class}</strong>
                          {predictionResult.shap_explanation.output_space && ` (Output Space: ${predictionResult.shap_explanation.output_space})`}
                        </p>
                      </div>

                      {predictionResult.shap_explanation.base_value !== null && predictionResult.shap_explanation.base_value !== undefined && (
                        <div className="text-right text-xs bg-slate-900 px-3 py-1.5 rounded-xl border border-slate-800">
                          <span className="text-slate-400 block text-[10px]">Model Expected Value (Base)</span>
                          <span className="font-mono font-bold text-indigo-400">
                            {predictionResult.shap_explanation.base_value}
                          </span>
                        </div>
                      )}
                    </div>

                    {predictionResult.shap_explanation.status === 'available' && predictionResult.shap_explanation.top_features.length > 0 ? (
                      <div className="space-y-3">
                        <div className="flex justify-between items-center text-[11px] text-slate-400 pb-1">
                          <span>Top Contributing Biomarkers (Ranked by Absolute SHAP Attribution)</span>
                          <span>Signed Impact on Predicted Tier</span>
                        </div>

                        {predictionResult.shap_explanation.top_features.map((feat, idx) => {
                          const isSupport = feat.direction === 'supports';
                          const absVal = Math.abs(feat.shap_value);
                          const barWidth = Math.min(100, Math.max(12, (absVal / 0.4) * 100));

                          return (
                            <div key={idx} className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-3.5 space-y-2">
                              <div className="flex items-center justify-between text-xs">
                                <div className="flex items-center space-x-2">
                                  <span className="font-semibold text-slate-200">{feat.display_name}</span>
                                  <span className="px-2 py-0.5 rounded-md text-[11px] font-mono bg-slate-800 text-slate-300">
                                    {feat.patient_value}
                                  </span>
                                </div>
                                <span className={`font-mono font-bold ${isSupport ? 'text-emerald-400' : 'text-rose-400'}`}>
                                  {feat.shap_value > 0 ? '+' : ''}{feat.shap_value} SHAP
                                </span>
                              </div>

                              {/* Attributed Impact Bar */}
                              <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden flex">
                                <div
                                  className={`h-full rounded-full transition-all duration-700 ${
                                    isSupport ? 'bg-gradient-to-r from-teal-500 to-emerald-400' : 'bg-gradient-to-r from-rose-500 to-amber-500'
                                  }`}
                                  style={{ width: `${barWidth}%` }}
                                />
                              </div>

                              <div className="flex items-center justify-between text-[11px]">
                                <span className="text-slate-400">{feat.clinical_insight}</span>
                                <span
                                  className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${
                                    isSupport ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                                  }`}
                                >
                                  {feat.impact}
                                </span>
                              </div>
                            </div>
                          );
                        })}

                        {/* Additivity & Omitted Features Explanation */}
                        {predictionResult.shap_explanation.omitted_features_shap_sum !== null && (
                          <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800 text-[11px] text-slate-400 space-y-1">
                            <div className="flex justify-between font-mono">
                              <span>Omitted features net SHAP contribution:</span>
                              <span className="font-bold text-slate-200">
                                {predictionResult.shap_explanation.omitted_features_shap_sum > 0 ? '+' : ''}
                                {predictionResult.shap_explanation.omitted_features_shap_sum}
                              </span>
                            </div>
                            <p className="text-[10px] text-slate-500 leading-relaxed">
                              * Additivity Property: The explainer base value ({predictionResult.shap_explanation.base_value}) plus the sum of all individual feature attributions equals the model probability output ({(predictionResult.prediction.confidence * 100).toFixed(1)}%).
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="p-6 bg-slate-900/40 rounded-2xl border border-slate-800 text-center text-xs text-slate-400 space-y-1">
                        <p className="font-semibold text-slate-300">SHAP Explanations Unavailable</p>
                        <p className="text-[11px] text-slate-500">
                          {predictionResult.shap_explanation.reason || 'Explanations require an active machine learning model and loaded SHAP explainer.'}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Recommendations */}
                  <div className="bg-slate-950/90 border border-slate-800 rounded-3xl p-6 shadow-xl backdrop-blur">
                    <div className="flex items-center space-x-2 mb-4">
                      <Icons.HeartPulse />
                      <div>
                        <h3 className="text-base font-semibold text-white">Educational Preventive Guidance</h3>
                        <p className="text-xs text-slate-400">Standard ADA prevention and lifestyle guidelines</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                      {predictionResult.clinical_recommendations.map((rec, i) => (
                        <div key={i} className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded-md">
                              {rec.category}
                            </span>
                            <span
                              className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                                rec.urgency === 'Immediate'
                                  ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                                  : rec.urgency === 'Moderate'
                                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                                  : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                              }`}
                            >
                              {rec.urgency}
                            </span>
                          </div>
                          <h4 className="text-xs font-bold text-slate-100">{rec.title}</h4>
                          <p className="text-xs text-slate-400 leading-relaxed">{rec.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                /* Un-evaluated Initial State */
                <div className="bg-slate-950/60 border border-slate-800 rounded-3xl p-12 text-center text-slate-400 space-y-3">
                  <div className="p-3 bg-slate-900 w-12 h-12 rounded-2xl mx-auto flex items-center justify-center text-indigo-400">
                    <Icons.Activity />
                  </div>
                  <h3 className="text-base font-semibold text-slate-200">Ready for Risk Assessment</h3>
                  <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
                    Select a synthetic scenario above or adjust clinical laboratory biomarkers and vitals on the left, then click <strong>Run Assessment</strong>.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* TAB 2: Model Validation & Comparative Baselines */
          <div className="space-y-6">
            <div className="bg-slate-950/80 border border-slate-800 rounded-3xl p-6 shadow-xl backdrop-blur space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-slate-800">
                <div>
                  <h2 className="text-lg font-bold text-white">Model Evaluation & Comparative Baselines</h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Evaluated on 20% hold-out test partition (n=800) from synthetic benchmark cohort
                  </p>
                </div>
                <div className="text-xs text-amber-400 bg-amber-500/10 border border-amber-500/30 px-3 py-1.5 rounded-xl">
                  Dataset: <strong>Synthetic Benchmark Cohort</strong> (Not real-world patients)
                </div>
              </div>

              {metricsError ? (
                <div className="p-8 text-center text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-2xl space-y-2">
                  <p className="font-bold text-sm">Evaluation Unavailable</p>
                  <p className="text-xs text-rose-200">{metricsError}</p>
                  <p className="text-[11px] text-slate-400">
                    Run <code className="bg-slate-900 px-2 py-0.5 rounded text-indigo-300">python ml_pipeline.py</code> in terminal to generate metrics.
                  </p>
                </div>
              ) : metricsData ? (
                <>
                  {/* Primary Metric Score Cards */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: 'Overall Accuracy', value: `${(metricsData.accuracy * 100).toFixed(2)}%`, sub: 'Hold-out test sample' },
                      { label: 'F1-Score (Macro)', value: metricsData.f1_macro.toFixed(4), sub: 'Unweighted class mean' },
                      { label: 'Macro ROC-AUC', value: metricsData.roc_auc_macro.toFixed(4), sub: 'One-vs-Rest (OvR)' },
                      { label: 'Test Log-Loss', value: metricsData.calibration_log_loss !== null ? metricsData.calibration_log_loss.toFixed(4) : 'N/A', sub: 'Probability calibration' },
                    ].map((card, i) => (
                      <div key={i} className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4">
                        <span className="text-xs text-slate-400 block">{card.label}</span>
                        <span className="text-2xl font-black text-white block my-1">{card.value}</span>
                        <span className="text-[11px] text-slate-500 block">{card.sub}</span>
                      </div>
                    ))}
                  </div>

                  {/* Comparative Baselines Table */}
                  {metricsData.baselines_comparison && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-semibold text-slate-200">
                        Comparative Model Baselines (Evaluated on Same Test Split)
                      </h3>
                      <div className="overflow-x-auto rounded-2xl border border-slate-800">
                        <table className="w-full text-xs text-left text-slate-300">
                          <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 text-[11px] uppercase">
                            <tr>
                              <th className="px-4 py-3">Estimator / Approach</th>
                              <th className="px-4 py-3">Type</th>
                              <th className="px-4 py-3 text-center">Accuracy</th>
                              <th className="px-4 py-3 text-center">F1 (Macro)</th>
                              <th className="px-4 py-3">Description</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800">
                            <tr className="hover:bg-slate-900/40">
                              <td className="px-4 py-3 font-semibold text-indigo-300">Random Forest (Tuned)</td>
                              <td className="px-4 py-3 text-slate-400">Tree Ensemble</td>
                              <td className="px-4 py-3 text-center font-mono font-bold text-emerald-400">
                                {(metricsData.accuracy * 100).toFixed(2)}%
                              </td>
                              <td className="px-4 py-3 text-center font-mono font-bold text-emerald-400">
                                {metricsData.f1_macro.toFixed(4)}
                              </td>
                              <td className="px-4 py-3 text-slate-400 text-[11px]">Primary balanced ensemble classifier.</td>
                            </tr>
                            {Object.entries(metricsData.baselines_comparison).map(([key, b]) => (
                              <tr key={key} className="hover:bg-slate-900/40">
                                <td className="px-4 py-3 font-medium text-slate-200">{b.model}</td>
                                <td className="px-4 py-3 text-slate-400">Baseline</td>
                                <td className="px-4 py-3 text-center font-mono text-slate-300">
                                  {(b.accuracy * 100).toFixed(2)}%
                                </td>
                                <td className="px-4 py-3 text-center font-mono text-slate-300">{b.f1_macro.toFixed(4)}</td>
                                <td className="px-4 py-3 text-slate-400 text-[11px]">{b.description}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Multi-Class Confusion Matrix */}
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-slate-200">
                      Multi-Class Confusion Matrix (Actual vs Predicted)
                    </h3>
                    <div className="overflow-x-auto rounded-2xl border border-slate-800">
                      <table className="w-full text-xs text-left text-slate-300">
                        <thead className="bg-slate-900 text-slate-400 border-b border-slate-800 text-[11px] uppercase">
                          <tr>
                            <th className="px-4 py-3">Actual Class \ Predicted</th>
                            <th className="px-4 py-3 text-center">Healthy</th>
                            <th className="px-4 py-3 text-center">Pre-diabetic</th>
                            <th className="px-4 py-3 text-center">Diabetic</th>
                            <th className="px-4 py-3 text-right">Sensitivity (Recall)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                          {metricsData.confusion_matrix.classes.map((actualName, rowIdx) => {
                            const row = metricsData.confusion_matrix.matrix[rowIdx];
                            const total = row.reduce((a, b) => a + b, 0);
                            const correct = row[rowIdx];
                            const sens = total > 0 ? ((correct / total) * 100).toFixed(1) : '0.0';

                            return (
                              <tr key={actualName} className="hover:bg-slate-900/40">
                                <td className="px-4 py-3 font-semibold text-slate-200">{actualName}</td>
                                {row.map((cellCount, colIdx) => (
                                  <td
                                    key={colIdx}
                                    className={`px-4 py-3 text-center font-mono ${
                                      rowIdx === colIdx ? 'bg-indigo-500/15 text-indigo-300 font-bold' : 'text-slate-400'
                                    }`}
                                  >
                                    {cellCount}
                                  </td>
                                ))}
                                <td className="px-4 py-3 text-right font-mono text-emerald-400 font-semibold">
                                  {sens}% ({correct}/{total})
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              ) : (
                <div className="p-8 text-center text-slate-400">Loading evaluation metrics...</div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500 mt-12 space-y-1">
        <p>EndocrineAI™ Educational Machine Learning Prototype · Evaluated on Synthetic Benchmark Data</p>
        <p className="text-[11px] text-slate-600">
          Not intended for real-world clinical diagnosis or treatment planning. Classifies current diagnostic categories without longitudinal progression tracking.
        </p>
      </footer>
    </div>
  );
}
