我现在有一个需要求解的半定规划SDP：
目标函数：
$$G_{x^{*},y^{*}}^{MDI}=\operatorname* {max}_{\{\tilde{M}_{a,b,e}^{A_0B_0}\}_{a,b,e}} \ \text{Tr}\left( \sum _{e}\tilde{M}_{a,b,e=(a,b)}^{A_0B_0} \left( \psi _{x^{*}}^{A_0} \otimes \psi _{y^{*}}^{B_0} \right) \right)$$

约束条件（s.t.）：
1. 与观测数据一致：
$$\text{Tr}\left( \sum_{e} \tilde{M}_{a,b,e}^{A_0B_0} \left( \psi_{x}^{A_0} \otimes \psi_{y}^{B_0} \right) \right) = p\left(a, b | \psi_{x}, \psi_{y}\right), \quad \forall x, y, a, b$$
2. 联合POVM算子非负：
$$\tilde{M}_{a, b, e}^{A_{0} B_{0}} \succeq 0, \quad \forall a, b, e$$
3. 无信号约束（对Alice的输出求和）：
$$\sum_{a} \tilde{M}_{a, b, e}^{A_{0} B_{0}}=\mathbb{I}^{A_{0}} \otimes \tilde{M}_{b, e}^{B_{0}}, \quad \forall b, e$$
4. 无信号约束（对Bob的输出求和）：
$$\sum_{b} \tilde{M}_{a, b, e}^{A_{0} B_{0}}=\tilde{M}_{a, e}^{A_{0}} \otimes \mathbb{I}^{B_{0}}, \quad \forall a, e$$
5. Bob局部POVM归一化：
$$\sum_{b} \tilde{M}_{b, e}^{B_{0}}=p(e) \mathbb{I}^{B_{0}}, \quad \forall e$$
6. Alice局部POVM归一化：
$$\sum_{a} \tilde{M}_{a, e}^{A_{0}}=p(e) \mathbb{I}^{A_{0}}, \quad \forall e$$
7. Eve结果概率归一化：
$$\sum_{e} p(e)=1$$