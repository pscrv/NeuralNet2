using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public class EmbeddingBatch : VectorBatch
    {
        #region constructors
        public EmbeddingBatch(MatrixIdentifier id) 
            : base(id)
        { }
        #endregion


        #region properties
        public override int Dimension { get { return AlgebraManager.GetRowCount(MatrixID); } }
        #endregion
    }
}
