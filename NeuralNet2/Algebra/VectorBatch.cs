using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public class VectorBatch : NetMatrix
    {
        #region constructors
        public VectorBatch(MatrixIdentifier id) 
            : base(id)
        { }
        #endregion


        #region properties
        public virtual int VectorCount { get { return AlgebraManager.GetRowCount(MatrixID); } }
        public virtual int Dimension { get { return AlgebraManager.GetColumnCount(MatrixID); } }
        #endregion
    }
}
