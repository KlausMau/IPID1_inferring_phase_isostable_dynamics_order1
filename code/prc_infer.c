#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<Eigen/Dense>
#define Pi 3.14159

using namespace Eigen;

double *import_file(int *cnt, char *filename){
	FILE *f;
	f = fopen(filename,"rt");
	*cnt = 0;
	double dp;
	while(fscanf(f, "%lf\n", &dp) != EOF) (*cnt)++;
	rewind(f);
	double *data = (double*) malloc(*cnt*sizeof(double));
	for(int i = 0; i < *cnt; ++i) fscanf(f, "%lf\n", &data[i]);
	fclose(f);
	return data;
}

double PRC_from_sol(double fi, VectorXf sol, int N_fourier){
	double res = sol(1);
	for(int n = 1; n < N_fourier; ++n){
		res += sol(2*n)*cos(n*fi);
		res += sol(2*n+1)*sin(n*fi);
	}
	return res;
}

int main(int argc, char *argv[]){
	
	char *str = argv[1];
	int N_fourier = atoi(str);
	
	// import events, phase and forcing
	int events_cnt, phase_cnt, forcing_cnt;
	char filename_e[] = "events/events.txt";
	double *events = import_file(&events_cnt, filename_e);
	char filename_p[] = "phase/phase.txt";
	double *phase = import_file(&phase_cnt, filename_p);
	char filename_f[] = "forcing/forcing.txt";
	double *forcing = import_file(&forcing_cnt, filename_f);
	
	// fill the matrix A
	MatrixXf A(events_cnt-1, 1+1+2*N_fourier); 
	for(int i = 0; i < events_cnt-1; ++i){
		A(i,0) = events[i+1]-events[i];
		// Fourier integrals
		A(i,1) = forcing[int(ceil(events[i]))-1]*(ceil(events[i])-events[i]); // first fractional timestep
		for(int t = ceil(events[i]); t < floor(events[i+1]); ++t){
			A(i,1) += forcing[t];
		}
		A(i,1) += forcing[int(floor(events[i+1]))]*(events[i+1]-floor(events[i+1])); // last fractional timestep
		for(int n = 1; n <= N_fourier; ++n){
			// cos
			A(i,2*n) = forcing[int(ceil(events[i]))-1]*cos(n*phase[int(ceil(events[i]))-1])*(ceil(events[i])-events[i]); // first fractional timestep
			for(int t = ceil(events[i]); t < floor(events[i+1]); ++t){
				A(i,2*n) += forcing[t]*cos(n*phase[t]);
			}
			A(i,2*n) += forcing[int(floor(events[i+1]))]*cos(n*phase[int(floor(events[i+1]))])*(events[i+1]-floor(events[i+1])); // last fractional timestep
			// sin
			A(i,2*n+1) = forcing[int(ceil(events[i]))-1]*sin(n*phase[int(ceil(events[i]))-1])*(ceil(events[i])-events[i]); // first fractional timestep
			for(int t = ceil(events[i]); t < floor(events[i+1]); ++t){
				A(i,2*n+1) += forcing[t]*sin(n*phase[t]);
			}
			A(i,2*n+1) += forcing[int(floor(events[i+1]))]*sin(n*phase[int(floor(events[i+1]))])*(events[i+1]-floor(events[i+1])); // last fractional timestep
		}
	}
	// fill the vector b
	VectorXf b(events_cnt-1);
	for(int i = 0; i < events_cnt-1; ++i) b(i) = 2*Pi;
	// minimization
	MatrixXf ATAA = A.transpose()*A;
	ATAA = ATAA.inverse();
	ATAA = ATAA*A.transpose();
	VectorXf sol(1+1+2*N_fourier);
	sol = ATAA*b;

	// write on file
	FILE *f; 
	f = fopen("PRC/sol.txt", "wt");
	for(int i = 0; i < 1+1+2*N_fourier; ++i){
		fprintf(f, "%lf\n", sol(i));
	}
	fclose(f);
	
	// phase recalculation
	double *new_phase = new double[forcing_cnt];
	double *psis = new double[events_cnt];
	for(int t = 0; t < floor(events[0])+1; ++t) new_phase[t] = -1;
	for(int i = 0; i < events_cnt-1; ++i){
		double phase = (sol(0)+PRC_from_sol(0,sol,N_fourier)*forcing[int(ceil(events[i]))-1])*(int(ceil(events[i]))-events[i]); // first fractional timestep
		new_phase[int(ceil(events[i]))] = phase;
		for(int t = int(ceil(events[i]))+1; t < int(floor(events[i+1]))+1; ++t){
			phase = phase + sol(0) + PRC_from_sol(phase,sol,N_fourier)*forcing[t];
			new_phase[t] = phase;
		}
		// phase at the end
		double psi = phase + (sol(0)+PRC_from_sol(phase,sol,N_fourier)*forcing[int(floor(events[i+1]))])*(events[i+1]-int(floor(events[i+1]))); // last fractional timestep
		psis[i] = psi;
		// rescale so its 2pi at the end
		for(int t = int(ceil(events[i])); t < int(floor(events[i+1]))+1; ++t){
			new_phase[t] = 2*Pi*new_phase[t]/psi;
		}
	}
	
	// write on file
	f = fopen("phase/phase.txt", "wt");
	for(int i = 0; i < forcing_cnt; ++i){
		fprintf(f, "%lf\n", new_phase[i]);
	}
	fclose(f);
	
	// error
	double var = 0;
	for(int i = 0; i < events_cnt-1; ++i){
		var += (psis[i]-2*Pi)*(psis[i]-2*Pi);
	}
	var /= (events_cnt-1);
	double error = sqrt(var);
	// error 0
	double *Ts = new double[events_cnt-1];
	double avgT = 0;
	for(int i = 0; i < events_cnt-1; ++i){
		Ts[i] = events[i+1]-events[i];
		avgT += Ts[i];
	}
	avgT /= (events_cnt-1);
	double avg_w = 2*Pi/avgT;
	double var0 = 0;
	for(int i = 0; i < events_cnt-1; ++i){
		var0 += (avg_w*Ts[i]-2*Pi)*(avg_w*Ts[i]-2*Pi);
	}
	var0 /= (events_cnt-1);
	double error0 = sqrt(var0);
	
	// write on file
	f = fopen("error/error_prc.txt", "wt");
	fprintf(f, "%lf\n", error);
	fclose(f);
	f = fopen("error/error0_prc.txt", "wt");
	fprintf(f, "%lf\n", error0);
	fclose(f);

	return 0;
	
}
