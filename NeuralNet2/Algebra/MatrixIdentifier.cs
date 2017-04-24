using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public class MatrixIdentifier
    {
        private static int __id = 0;

        public int ID { get; private set; }

        public MatrixIdentifier()
        {
            ID = __id;
            __id++;
        }
    }
}
