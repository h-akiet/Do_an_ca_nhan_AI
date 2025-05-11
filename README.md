# Do_an_ca_nhan_AI-Nguyễn Hoàng Anh Kiệt-23110247
1.Mục tiêu  
- Hiểu rõ được bản chất và nguyên lí các thuật toán tìm kiếm AI
- Áp dụng được các thuật toán đã học vào vấn đề thực tế (áp dụng thuật toán, tạo giao diện cho trò chơi 8 Puzzle)
- So sánh, đánh giá ưu nhược điểm của từng (nhóm) thuật toán
- Có thể đưa ra giải pháp nâng cấp(nếu có)
2. Nội dung   
  2.1. Các thuật toán Tìm kiếm không có thông tin
     - Thành phần chính: ma trận đầu vào, ma trận đầu ra, thuật toán áp dụng, tập các hành động sinh trạng thái mới
     - Solution: tập các trạng thái mới được sinh ra đại diện cho từng bước giải trò chơi
> 📊 Thuật toán BFS: tìm kiếm bằng cách duyệt theo chiều rộng(duyệt từng lớp) các trạng thái được sinh ra
> <p align="center">
>   <img src="assets/BFS.gif" width="500"/>
> </p>
> 📊 Thuật toán DFS: duyệt theo chiều sâu, duyệt cho tới cuối nhánh của không gian trạng thái
> <p align="center">
>   <img src="assets/DFS.gif" width="500"/>
> </p>
> 📊 Thuật toán IDS: sử dụng DfS nhưng có giới hạn dộ sâu -> tối ưu hơn về mặc thời gian và hiệu suất
> <p align="center">
>   <img src="assets/IDS.gif" width="500"/>
> </p>
> 📊 Thuật toán UCS: duyệt theo tổng chi phí đường đi (là 1 tương ứng với mỗi step
> <p align="center">
>   <img src="assets/UCS.gif" width="500"/>
> </p>  
    2.2. Các thuật toán Tìm kiếm có thông tin  
     - Thành phần chính: ma trận đầu vào, ma trận đầu ra, thuật toán áp dụng, tập các hành động sinh trạng thái mới
     - Solution: tập các trạng thái mới được sinh ra đại diện cho từng bước giải trò chơi
>  📊 Thuật toán GREEDY: dựa vào chi phí Heurictic (cụ thể là Manhatan được dùng trong mã nguồn) thấp nhất để đưa ra hành động
> <p align="center">
>   <img src="assets/greedy.gif" width="500"/>
> </p> 
>  📊 Thuật toán A*: dựa vào chi phí g(n): chi phí thực tế (tương ứng với step trong trạng thái hiện tại của curent_state) và h(n): heurictic để xác định trạng thái cần mở rộng
> <p align="center">
>   <img src="assets/A_star.gif" width="500"/>
> </p>  
> 📊 Thuật toán IDA*: tương tự A*, nhưng có thiết lập chiều sâu, tối ưu hơn về vùng nhớ, hiệu năng tìm kiếm, thuật toán sẽ luôn tìm thấy lời giải nếu có tồn tại
> <p align="center">
>   <img src="assets/IDA_star.gif" width="500"/>
> </p>  
    2.3. Các thuật toán Tìm kiếm cục bộ  
> 📊 Thuật toán SIMPLE HILL
> <p align="center">
>   <img src="assets/simple.gif" width="500"/>
> </p>  
> 📊 Thuật toán STEPEST HILL
> <p align="center">
>   <img src="assets/stepest.gif" width="500"/>
> </p> 
>  📊 Thuật toán STOCHASTIC HILL
> <p align="center">
>   <img src="assets/stochastic.gif" width="500"/>
> </p>  
>  📊 Thuật toán SIMULATED_ANNEALING
> <p align="center">
>   <img src="assets/S_A.gif" width="500"/>
> </p> 
>  📊 Thuật toán BEAM Search
> <p align="center">
>   <img src="assets/beam.gif" width="500"/>
> </p>  
>  📊 Thuật toán GENETIC
> <p align="center">
>   <img src="assets/Genetic.gif" width="500"/>
> </p> 
    2.4. Các thuật toán Tìm kiếm trong môi trường phức tạp  
>  📊 Thuật toán Non Observation
> <p align="center">
>   <img src="assets/non.gif" width="500"/>
> </p>  
>  📊 Thuật toán Partial Observation
> <p align="center">
>   <img src="assets/partial.gif" width="500"/>
> </p> 
    2.5. Các thuật toán Tìm kiếm có ràng buộc  
>  📊 Thuật toán Backtracking
> <p align="center">
>   <img src="assets/backtracking.gif" width="500"/>
> </p>   
>  📊 Thuật toán GenerateE & Test
> <p align="center">
>   <img src="assets/test.gif" width="500"/>
> </p>  
>  📊 Thuật toán AC-3
> <p align="center">
>   <img src="assets/ac3.gif" width="500"/>
> </p> 
    



