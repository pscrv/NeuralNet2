namespace fsh_net

[<AutoOpen>]
module BatchFunctions =

    let MakeBatch (data: TrainingData) index batchsize = 

        let batchlength = data.Input.ColumnCount
        let start_index = (index * batchsize) %  batchlength
        let end_index_nominal = start_index + batchsize
        let end_index =
            if end_index_nominal > batchlength
            then end_index_nominal % batchlength
            else end_index_nominal

        if start_index < end_index 
            then
                {
                    Name = "Batch"
                    Input = data.Input.SubMatrix (0, data.Input.RowCount, start_index, batchsize );
                    Target = data.Target.SubMatrix (0, data.Target.RowCount, start_index, batchsize)
                }
            else
                let left_size = batchlength - start_index
                let right_size = batchsize - left_size
                let left_input_matrix = data.Input.SubMatrix (0, data.Input.RowCount, start_index, left_size )
                let left_target_matrix = data.Target.SubMatrix (0, data.Target.RowCount, start_index, left_size )
                let right_input_matrix = data.Input.SubMatrix(0, data.Input.RowCount, 0, right_size)
                let right_target_matrix = data.Target.SubMatrix(0, data.Target.RowCount, 0, right_size)
                {
                    Name = "Batch";
                    Input = left_input_matrix.Append(right_input_matrix);
                    Target = left_target_matrix.Append(right_target_matrix);
                }

