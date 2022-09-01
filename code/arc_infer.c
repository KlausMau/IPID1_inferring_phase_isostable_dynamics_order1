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

double ARC_from_sol(double fi, VectorXf sol, int N_fourier){
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
	int events_cnt, a_events_cnt, amplitude_cnt, phase_cnt, forcing_cnt;
	char filename_e[] = "events/phase_thr_events.txt";
	double *events = import_file(&events_cnt, filename_e);
	char filename_ae[] = "events/amplitude_x0_at_events.txt";
	double *a_events = import_file(&a_events_cnt, filename_ae);
	char filename_a[] = "amplitude/amplitude_x0.txt";
	double *amplitude = import_file(&amplitude_cnt, filename_a);
	char filename_p[] = "phase/phase.txt";
	double *phase = import_file(&phase_cnt, filename_p);
	char filename_f[] = "forcing/forcing.txt";
	double *forcing = import_file(&forcing_cnt, filename_f);
	
	// fill the matrix A
	MatrixXf A(events_cnt-1, 1+1+2*N_fourier+1); 
	for(int i = 0; i < events_cnt-1; ++i){
		// Floquet exponent
		A(i,0) = -amplitude[int(ceil(events[i]))-1]*(ceil(events[i])-events[i]); // first fractional timestep
		for(int t = ceil(events[i]); t < floor(events[i+1]); ++t){
			A(i,0) -= amplitude[t];
		}
		A(i,0) -= amplitude[int(floor(events[i+1]))]*(events[i+1]-floor(events[i+1])); // last fractional timestep
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
		// the k*x0 integral for floquet exponent times x0
		A(i,2*N_fourier+2) = ceil(events[i])-events[i]; // first fractional timestep
		for(int t = ceil(events[i]); t < floor(events[i+1]); ++t){
			A(i,2*N_fourier+2) += 1;
		}
		A(i,2*N_fourier+2) += events[i+1]-floor(events[i+1]); // last fractional step	
	}
	// fill the vector b
	VectorXf b(events_cnt-1);
	for(int i = 0; i < events_cnt-1; ++i) b(i) = a_events[i+1]-a_events[i];
	// minimization
	MatrixXf ATAA = A.transpose()*A;
	ATAA = ATAA.inverse();
	ATAA = ATAA*A.transpose();
	VectorXf sol(1+1+2*N_fourier+1);
	sol = ATAA*b;
	
	// write on file
	FILE *f; 
	f = fopen("ARC/sol.txt", "wt");
	for(int i = 0; i < 1+1+2*N_fourier+1; ++i){
		fprintf(f, "%lf\n", sol(i));
	}
	fclose(f);
	
	// estimate x0
	double x0 = sol[2+2*N_fourier]/sol[0]; // k*x0/k
	
	// amplitude recalculation
	double *new_amp = new double[forcing_cnt];
	double *psis = new double[events_cnt];
	for(int t = 0; t < floor(events[0])+1; ++t) new_amp[t] = -1;
	for(int i = 0; i < events_cnt-1; ++i){
		double amp = a_events[i]-x0 + (-sol[0]*(a_events[i]-x0) + ARC_from_sol(0,sol,N_fourier)*forcing[int(ceil(events[i]))-1])*(ceil(events[i])-events[i]); // first fractional timestep
		new_amp[int(ceil(events[i]))] = amp;
		for(int t = ceil(events[i])+1; t < int(floor(events[i+1]))+1; ++t){
			amp = amp + (-sol[0]*amp+ARC_from_sol(phase[t],sol,N_fourier)*forcing[t]);
			new_amp[t] = amp;
		}
		// amplitude at the end
		double psi = amp + (-sol[0]*amp+ARC_from_sol(phase[int(floor(events[i+1]))],sol,N_fourier)*forcing[int(floor(events[i+1]))])*(events[i+1]-floor(events[i+1])); // last fractional timestep
		psis[i] = psi;
		// no rescaling here, it can cause inaccuracies
	}

	// write on file
	f = fopen("amplitude/amplitude.txt", "wt");
	for(int i = 0; i < forcing_cnt; ++i){
		fprintf(f, "%lf\n", new_amp[i]);
	}
	fclose(f);
	f = fopen("amplitude/amplitude_x0.txt", "wt");
	for(int i = 0; i < forcing_cnt; ++i){
		fprintf(f, "%lf\n", new_amp[i]+x0);
	}
	fclose(f);
	
	// error
	double var = 0;
	for(int i = 0; i < events_cnt-1; ++i){
		var += (psis[i]-(a_events[i+1]-x0))*(psis[i]-(a_events[i+1]-x0));
	}
	var /= (a_events_cnt-1);
	double error = sqrt(var);
	// 0 error
	double var0 = 0;
	for(int i = 0; i < events_cnt-1; ++i){
		var0 += (a_events[i]-x0)*(a_events[i]-x0);
	}
	var0 /= (a_events_cnt-1);
	double error0 = sqrt(var0);
	
	// write on file
	f = fopen("error/error_arc.txt", "wt");
	fprintf(f, "%lf\n", error);
	fclose(f);
	f = fopen("error/error0_arc.txt", "wt");
	fprintf(f, "%lf\n", error0);
	fclose(f);
	
	return 0;
	
}
