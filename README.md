# protein_folding

## Tổng quát dự án

### Protein Folding là gì

Proteins are the biological molecules that "do" things, they are the molecular machines of biochemistry. Enzymes that break down food, hemoglobin that carries oxygen in blood, and actin filaments that make muscles contract are all proteins. They are made from long chains of amino acids, and the sequence of these chains is the information that is stored in DNA. However, its a large step to go from a 2D chain of amino acids to a 3D structure capable of working.

The process of this 2D structure folding on itself into a stable, 3D shape is called protein folding. For the most part, this process happens naturally and the end structure is in a much lower free energy state than the string. Like a bag of legos though, it is not enough to just know the building blocks being used, its the way they're supposed to be put together that matters. "Form defines function" is a common phrase in biochemsitry, and it is the quest to determine form, and thus function of proteins, that makes this process so important to understand and simulate.

### Mục tiêu

Sử dụng application OpenMM để chạy simulation cho các file PDB (Protein Data Bank) để folding.
Sau đó tìm điểm năng lượng thấp nhất có thể đạt được của hệ thống.

## Details

### Requirements

- Linux
- GPU
- Conda

### Preparation

```
bash install.sh
conda activate folding-protein
pip install -e .
```

### Running

```
python running_simulation.py {pdb_id}
python simulated_annealing.py {pdb_id}
```

**Lưu ý: List pdb_id hiện tại sẽ nằm ở trong filder miner_config**

Ví dụ
```
python running_simulation.py 3lil
```

### Detail các bước chạy simulation 

#### Input
Input đầu vào vào bao gồm 3 file (ví dụ trong folder miner-config)
- **1 file pkl**: Chứa config để chạy simulation. Config này cũng là config khởi tạo và cũng là config được sử dụng để validate simulation
- **1 file pdb**: chứa các protein, atom...
- **1 file em.cpt**: là file checkpoint sau khi chạy được 1000 steps với default config

#### Running simulation

Sau khi load simulation + checkpoint hiện tại ở file em.cpt, thực hiện chạy simulation. 

- Ví dụ với default simulation (code ở file running_simulation.py)

```
simulation.loadCheckpoint(cpt_file_mapper(output_dir, state))
simulation.step(steps_to_run)
```

- Ví dụ với simulation thêm kỹ thuật simulated annealing thay đổi nhiệt độ thì code sẽ như bên dưới 

```python
simulation.loadCheckpoint(cpt_file_mapper(output_dir, state))
start_time = time.time()
middle = (annealing_loop - 1) // 2

for i in range(annealing_loop):
    if i < middle:
        temperature_tmp = temperature_tmp + jump_temp
        integrator.setTemperature(temperature_tmp * unit.kelvin)
        simulation.step(CHECKPOINT_INTERVAL)
    elif i == middle:
        simulation.step(CHECKPOINT_INTERVAL * middle_cpt)
    else:
        temperature_tmp = temperature_tmp - jump_temp
        integrator.setTemperature(temperature_tmp * unit.kelvin)
        simulation.step(CHECKPOINT_INTERVAL)
simulation.step(steps_to_run)
```

**Toàn bộ phần chạy simulation này sẽ là phần custom cần phải viết với mục đích là tìm được điểm năng lượng thấp nhất của hệ thống**

#### Output file

Các output file sẽ nằm trong folder `data/{pdb_id}/validator`
Trong đó có các file cần lưu ý là 
- **md_0_1.cpt**: là file lưu checkpoint hiện tại sau khi chạy simulation
- **md_0_1.log**: file log chứa năng lượng ở từng step


#### Validate simulation

Phần code validate simulation sẽ check xem simulation chạy có hợp lệ hay không bằng việc **(hàm is_run_valid trong protein.py)**

- Load checkpoint hiện tại từ file md_0_1.cpt và **default config**
- Thực hiện chạy simulation thêm ít nhất 3000 steps, ghi log 3000 steps này ra  file `check_md_0_1.log`

- Check gradient 
Lấy 50 energy đầu tiên từ file log `check_md_0_1.log` sau đó kiểm tra mean của difference giữa các energy và đảm bảo năng lượng không thay đổi quá đột ngột

```python
def check_gradient(self, check_energies: np.ndarray) -> True:
    """This method checks the gradient of the potential energy within the first
    WINODW size of the check_energies array. Miners that return gradients that are too high,
    there is a *high* probability that they have not run the simulation as the validator specified.
    """
    WINDOW = 50  # Number of steps to calculate the gradient over
    GRADIENT_THRESHOLD = 10  # kJ/mol/nm

    mean_gradient = np.diff(check_energies[:WINDOW]).mean().item()
    return (
        mean_gradient <= GRADIENT_THRESHOLD
    )  # includes large negative gradients is passible
```

- Kiểm tra median_percent_diff giữa file `check_md_0_1.log` và file `md_0_1.log`

```python
percent_diff = abs(((check_energies - miner_energies) / miner_energies) * 100)
median_percent_diff = np.median(percent_diff)
```


**Energy cuối cùng nhận được sẽ là trung bình của 10 cuối cùng energies từ file check_md_0_1.log**

### Vấn đề đang gặp phải
Bảng dưới đây là 1 list những pdb và Expected Result mà khách hàng mong muốn.

Hiện tại bằng việc áp dùng simulated annealing (thay đổi nhiệt độ) thì Energy đạt được có giảm so với init energy tuy nhiên vẫn chưa đạt đến expected result. 

Vấn đề lớn nhất gặp phải là mặc dù có thể thay đổi khá nhiều config để giảm năng lượng hơn nữa, tuy nhiên khi đến phần validation thì lại không valid với 2 điều kiện trên


**Exptected Result**

| PDB  | Current Result | Expected Result |
|------|----------------|-----------------|
| 1ohp | -38449         | -41555          |
| 3lil | -6658          | -7927           |
| 2gud | -6160          | -7326           |
| 4grk | -20655         | -22481          |
| 5tkf | -39805         | -44522          |
||