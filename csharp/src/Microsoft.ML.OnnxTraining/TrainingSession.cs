// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// The TrainingSession manages the training process.
    /// </summary>
    public class TrainingSession : IDisposable
    {
        TrainingParameters m_param = new TrainingParameters();
        private OrtValueCollection m_expectedInputs = new OrtValueCollection(IntPtr.Zero);
        private OrtValueCollection m_expectedOutputs = new OrtValueCollection(IntPtr.Zero);
        bool _disposed = false;

        /// <summary>
        /// The constructor that creates a TrainingSession object.
        /// </summary>
        public TrainingSession()
        {
        }

        /// <summary>
        /// Finalizer. to cleanup training session in case it runs
        /// and the user forgets to Dispose() of the training session.
        /// </summary>
        ~TrainingSession()
        {
            Dispose(false);
        }

        #region Disposable

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // Suppress finalization.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            // dispose managed state (managed objects).
            if (disposing)
            {
                m_param.Dispose();
                m_expectedInputs.Dispose();
                m_expectedOutputs.Dispose();
            }

            _disposed = true;
        }

        #endregion

        /// <summary>
        /// Returns the TrainingParameters used for the training session.
        /// </summary>
        public TrainingParameters Parameters
        {
            get { return m_param; }
        }

        /// <summary>
        /// Initialize the training session using the OrtEnv.
        /// </summary>
        /// <param name="env">Specifies the OrtEnv to use.</param>
        public void Initialize(OrtEnv env)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtInitializeTraining(env.DangerousGetHandle(), m_param.DangerousGetHandle(), m_expectedInputs.DangerousGetHandle(), m_expectedOutputs.DangerousGetHandle()));

            m_param.ExpectedInputs = getTensorDefs(m_expectedInputs);
            m_param.ExpectedOutputs = getTensorDefs(m_expectedOutputs);
        }

        private List<KeyValuePair<string, List<int>>> getTensorDefs(OrtValueCollection col)
        {
            List<KeyValuePair<string, List<int>>> rgDefs = new List<KeyValuePair<string, List<int>>>();
            int nCount = col.Count;

            for (int i = 0; i < nCount; i++)
            {
                string strName;
                OrtValue val = col.GetAt(i, out strName);
                DisposableNamedOnnxValue shape = DisposableNamedOnnxValue.CreateTensorFromOnnxValue(strName, val);
                Tensor<long> shapet = shape.AsTensor<Int64>();
                List<int> rgShape = new List<int>();

                for (int j = 0; j < shapet.dimensions[0]; j++)
                {
                    long nDim = shapet.GetValue(j);
                    rgShape.Add((int)nDim);
                }

                rgDefs.Add(new KeyValuePair<string, List<int>>(strName, rgShape));
            }

            return rgDefs;
        }

        /// <summary>
        /// Run the training session.
        /// </summary>
        public void RunTraining()
        {           
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtRunTraining(m_param.DangerousGetHandle()));
        }

        /// <summary>
        /// End the training session.
        /// </summary>
        public void EndTraining()
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtEndTraining(m_param.DangerousGetHandle()));
        }
    }
}
